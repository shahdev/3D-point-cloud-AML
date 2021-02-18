import numpy as np
np.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(2)
import scipy.misc, scipy.io
import imageio
import time, os, sys
import threading
import util
from numpy import linalg as LA

print(util.toYellow("======================================================="))
print(util.toYellow("evaluate.py (evaluate/generate point cloud)"))
print(util.toYellow("======================================================="))

import tensorflow as tf
import data, graph, transform
import options

print(util.toMagenta("setting configurations..."))
opt = options.set(training=False)
# opt.batchSize = opt.inputViewN
opt.batchSize = 1
opt.chunkSize = 50

# create directories for evaluation output
util.mkdir("results_{0}/{1}".format(opt.group, opt.load))

print(util.toMagenta("building graph..."))
tf.reset_default_graph()

alpha_inp = opt.alpha_inp
alpha_flow = opt.alpha_flow
iter_ = 0
max_iters = 10000
attack_epsilon = opt.attack_epsilon/255
mu = 0.85
tau = opt.tau

with tf.device("/gpu:0"):
	VsPH = tf.placeholder(tf.float64, [None, 3])
	VtPH = tf.placeholder(tf.float64, [None, 3])
	_, minDist = util.projection(VsPH, VtPH)


# compute test error for one prediction
def computeTestError(Vs, Vt, type):
	VsN, VtN = len(Vs), len(Vt)
	if type == "pred->GT": evalN, VsBatchSize, VtBatchSize = min(VsN, 200), 200, 100000
	if type == "GT->pred": evalN, VsBatchSize, VtBatchSize = min(VsN, 200), 200, 40000
	# randomly sample 3D points to evaluate (for speed)
	randIdx = np.random.permutation(VsN)[:evalN]
	Vs_eval = Vs[randIdx]
	minDist_eval = np.ones([evalN]) * np.inf
	# for batches of source vertices
	VsBatchN = int(np.ceil(evalN / VsBatchSize))
	VtBatchN = int(np.ceil(VtN / VtBatchSize))
	for b in range(VsBatchN):
		VsBatch = Vs_eval[b * VsBatchSize:(b + 1) * VsBatchSize]
		minDist_batch = np.ones([len(VsBatch)]) * np.inf
		for b2 in range(VtBatchN):
			VtBatch = Vt[b2 * VtBatchSize:(b2 + 1) * VtBatchSize]
			md = sess.run(minDist, feed_dict={VsPH: VsBatch, VtPH: VtBatch})
			minDist_batch = np.minimum(minDist_batch, md)
		minDist_eval[b * VsBatchSize:(b + 1) * VsBatchSize] = minDist_batch
	return np.mean(minDist_eval)


# build graph
with tf.device("/gpu:0"):
	# ------ define input data ------
	inputImage_var = tf.Variable(np.expand_dims(np.load('target_image.npy'), axis=0), dtype=tf.float32, trainable=True)
	inputImage = tf.placeholder(tf.float32, shape=[opt.batchSize, opt.inH, opt.inW, 3])
	set_inputImage_op = inputImage_var.assign(inputImage)	

	renderTrans = tf.placeholder(tf.float32, shape=[opt.batchSize, opt.novelN, 4])
	depthGT = tf.placeholder(tf.float32, shape=[opt.batchSize, opt.novelN, opt.H, opt.W, 1])
	maskGT = tf.placeholder(tf.float32, shape=[opt.batchSize, opt.novelN, opt.H, opt.W, 1])

	PH = [renderTrans, depthGT, maskGT]
	# ------ build encoder-decoder ------
	encoder = graph.encoder if opt.arch == "original" else \
		graph.encoder_resnet if opt.arch == "resnet" else None
	decoder = graph.decoder if opt.arch == "original" else \
		graph.decoder_resnet if opt.arch == "resnet" else None
	latent = encoder(opt, inputImage_var)
	XYZ, maskLogit = decoder(opt, latent)  # [B,H,W,3V],[B,H,W,V]
	mask = tf.to_float(maskLogit > 0)
	# ------ build transformer ------
	fuseTrans = tf.nn.l2_normalize(opt.fuseTrans, dim=1)
	XYZid, ML = transform.fuse3D(opt, XYZ, maskLogit, fuseTrans)  # [B,1,VHW]

	newDepth, newMaskLogit, collision = transform.render2D(opt, XYZid, ML, renderTrans)  # [B,N,H,W,1]

	# ------ define loss ------
	loss_depth = graph.masked_l1_loss(newDepth - depthGT, tf.equal(collision, 1)) / (opt.batchSize * opt.novelN)
	loss_mask = graph.cross_entropy_loss(newMaskLogit, maskGT) / (opt.batchSize * opt.novelN)

	loss = loss_mask + opt.lambdaDepth * loss_depth
	train_op = tf.train.AdamOptimizer(learning_rate=alpha_inp).minimize(loss, var_list=[inputImage_var])

# load data
print(util.toMagenta("loading dataset..."))
dataloader = data.Loader(opt, loadNovel=False, loadTest=True)
CADN = len(dataloader.CADs)
chunkN = int(np.ceil(CADN / opt.chunkSize))
dataloader.loadChunk(opt, loadRange=[0, opt.chunkSize])

# prepare model saver/summary writer
saver = tf.train.Saver(tf.trainable_variables()[1:])
print(util.toYellow("======= EVALUATION START ======="))
timeStart = time.time()
# start session

source_img_path = 'source_image.npy'
target_img_path = 'target_image.npy'
target_renderTrans_path = 'target_renderTrans.npy'
target_depthGT_path = 'target_depthGT.npy'
target_maskGT_path = 'target_maskGT.npy'

def attack(sess):
	source_img = np.load(source_img_path)
	target_img = np.load(target_img_path)
	target_renderTrans = np.load(target_renderTrans_path)
	target_depthGT = np.load(target_depthGT_path)
	target_maskGT = np.load(target_maskGT_path)

	source_img = np.expand_dims(source_img, axis=0)
	target_img = np.expand_dims(target_img, axis=0)
	target_renderTrans = np.expand_dims(target_renderTrans, axis=0)
	target_depthGT = np.expand_dims(target_depthGT, axis=0)
	target_maskGT = np.expand_dims(target_maskGT, axis=0)

	runList = [XYZid, ML, loss]
	target_batch = {renderTrans: target_renderTrans, depthGT: target_depthGT, maskGT: target_maskGT}
	#sess.run(set_inputImage_op, feed_dict={inputImage: target_img})
	xyz, ml, l = sess.run(runList, feed_dict=target_batch)
	target_points = np.zeros([opt.batchSize, 1], dtype=np.object)
	for a in range(opt.batchSize):
		xyz1 = xyz[a].T  # [VHW,3]
		ml1 = ml[a].reshape([-1])  # [VHW]
		target_points[a, 0] = xyz1[ml1 > 0]
	import pdb; pdb.set_trace()
	pgd_max = source_img + attack_epsilon
	pgd_min = source_img - attack_epsilon
	x_adv = source_img.copy()
	global iter_, alpha_inp, alpha_flow, tau
	while iter_ < max_iters:
		adv_batch = {renderTrans: target_renderTrans, depthGT: target_depthGT,
					 maskGT: target_maskGT}
		sess.run(set_inputImage_op, feed_dict={inputImage: x_adv})
		xyz, ml, l = sess.run(runList, feed_dict=adv_batch)
		#sess.run(train_op, feed_dict=adv_batch)
		Vpred = np.zeros([opt.batchSize, 1], dtype=np.object)
		for a in range(opt.batchSize):
			xyz1 = xyz[a].T  # [VHW,3]
			ml1 = ml[a].reshape([-1])  # [VHW]
			Vpred[a, 0] = xyz1[ml1 > 0]
		pred2GT = computeTestError(Vpred[0][0], target_points[0][0], type="pred->GT") * 100
		GT2pred = computeTestError(target_points[0][0], Vpred[0][0], type="GT->pred") * 100
		#print(iter_, l, "pred2GT:", pred2GT, "GT2pred:", GT2pred, np.sum(np.power(x_adv - source_img,2)), np.max(flow_adv), np.min(flow_adv), flush=True)
		print(iter_, l, "pred2GT:", pred2GT, "GT2pred:", GT2pred, flush=True)
		iter_ += 1

		if iter_ % 1000 == 499:
			alpha_inp *= 1.2
		# tau *= 0.8
		if iter_ % 500 == 499:
			np.save('%s/adv_%d.npy' % (opt.save_dir, iter_), x_adv)

			xyz, ml, _, _ = sess.run(runList, feed_dict=adv_batch)

			Vpred = np.zeros([opt.batchSize, 1], dtype=np.object)
			for a in range(opt.batchSize):
				xyz1 = xyz[a].T  # [VHW,3]
				ml1 = ml[a].reshape([-1])  # [VHW]
				Vpred[a, 0] = xyz1[ml1 > 0]
			np.save('%s/points_%d.npy' % (opt.save_dir, iter_), Vpred[0][0])

			for image_index in range(adv_img.shape[0]):
				imageio.imwrite('%s/adv_image_%d_%d.png' % (opt.save_dir, image_index, iter_), (x_adv[image_index]* 255).astype(np.uint8))

tfConfig = tf.ConfigProto(allow_soft_placement=True)
tfConfig.gpu_options.allow_growth = True

with tf.Session(config=tfConfig) as sess:
		sess.run(tf.global_variables_initializer())
		util.restoreModel(opt, sess, saver)
		print(util.toMagenta("loading pretrained ({0})...".format(opt.load)))

		attack(sess)

print(util.toYellow("======= EVALUATION DONE ======="))


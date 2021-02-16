import numpy as np
np.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(2)
import scipy.misc, scipy.io
import time, os, sys
import threading
import util
from numpy import linalg as LA
import imageio
print(util.toYellow("======================================================="))
print(util.toYellow("evaluate.py (evaluate/generate point cloud)"))
print(util.toYellow("======================================================="))

import tensorflow as tf
import data, graph, transform
import options

import scipy.misc
print(util.toMagenta("setting configurations..."))
opt = options.set(training=False)
# opt.batchSize = opt.inputViewN
opt.batchSize = 50
opt.chunkSize = 50
attack_epsilon = 1.0 #8.0 / 255
threshold = 0.4
tau = opt.tau

# create directories for evaluation output
util.mkdir("results_{0}/{1}".format(opt.group, opt.load))

print(util.toMagenta("building graph..."))
tf.reset_default_graph()

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


def flow_st(images, flows, data_format='NHWC'):
	"""Flow-based spatial transformation of images.
	See Eq. (1) in Xiao et al. (arXiv:1801.02612).

	Args:
		images (tf.Tensor): images of shape `(B, H, W, C)` or `(B, C, H, W)`
							depending on `data_format`.
		flows (tf.Tensor): flows of shape `(B, 2, H, W)`, where the second
						   dimension indicates the dimension on which the pixel
						   shift is applied.
		data_format (str): ``'NHWC'`` or ``'NCHW'`` depending on the format of
						   the input images and the desired output.
	Returns:
		 `tf.Tensor` of the same shape and type as `images`.
	"""
	if data_format == 'NHWC':
		i_H = 1
	elif data_format == 'NCHW':
		i_H = 2
	else:
		raise ValueError("Provided data_format is not valid.")
	with tf.variable_scope('flow_st'):
		images_shape = tf.shape(images)
		flows_shape = tf.shape(flows)

		batch_size = images_shape[0]
		H = images_shape[i_H]
		W = images_shape[i_H + 1]

		# make sure that the input images and flows have consistent shape
		with tf.control_dependencies(
				[tf.assert_equal(
					tf.identity(images_shape[i_H:i_H + 2], name='images_shape_HW'),
					tf.identity(flows_shape[2:], name='flows_shape_HW')
				)]
		):
			# cast the input to float32 for consistency with the rest
			images = tf.cast(images, 'float32', name='images_float32')
			flows = tf.cast(flows, 'float32', name='flows_float32')

			if data_format == 'NCHW':
				images = tf.transpose(images, [0, 2, 3, 1])

			# basic grid: tensor with shape (2, H, W) with value indicating the
			# pixel shift in the x-axis or y-axis dimension with respect to the
			# original images for the pixel (2, H, W) in the output images,
			# before applying the flow transforms
			basegrid = tf.stack(
				tf.meshgrid(tf.range(H), tf.range(W), indexing='ij')
			)

			# go from (2, H, W) tensors to (B, 2, H, W) tensors with simple copy
			# across batch dimension
			batched_basegrid = tf.tile([basegrid], [batch_size, 1, 1, 1])

			# sampling grid is base grid + input flows
			sampling_grid = tf.cast(batched_basegrid, 'float32') + flows

			# separate shifts in x and y is easier--also we clip to the
			# boundaries of the image
			sampling_grid_x = tf.clip_by_value(
				sampling_grid[:, 1], 0., tf.cast(W - 1, 'float32')
			)
			sampling_grid_y = tf.clip_by_value(
				sampling_grid[:, 0], 0., tf.cast(H - 1, 'float32')
			)

			# now we need to interpolate

			# grab 4 nearest corner points for each (x_i, y_i)
			# i.e. we need a square around the point of interest
			x0 = tf.cast(tf.floor(sampling_grid_x), 'int32')
			x1 = x0 + 1
			y0 = tf.cast(tf.floor(sampling_grid_y), 'int32')
			y1 = y0 + 1

			# clip to range [0, H/W] to not violate image boundaries
			# - 2 for x0 and y0 helps avoiding black borders
			# (forces to interpolate between different points)
			x0 = tf.clip_by_value(x0, 0, W - 2, name='x0')
			x1 = tf.clip_by_value(x1, 0, W - 1, name='x1')
			y0 = tf.clip_by_value(y0, 0, H - 2, name='y0')
			y1 = tf.clip_by_value(y1, 0, H - 1, name='y1')

			# b is a (B, H, W) tensor with (B, H, W) = B for all (H, W)
			b = tf.tile(
				tf.reshape(
					tf.range(0, batch_size), (batch_size, 1, 1)
				),
				(1, H, W)
			)

			# get pixel value at corner coordinates
			# we stay indices along the last dimension and gather slices
			# given indices
			# the output is of shape (B, H, W, C)
			Ia = tf.gather_nd(images, tf.stack([b, y0, x0], 3), name='Ia')
			Ib = tf.gather_nd(images, tf.stack([b, y1, x0], 3), name='Ib')
			Ic = tf.gather_nd(images, tf.stack([b, y0, x1], 3), name='Ic')
			Id = tf.gather_nd(images, tf.stack([b, y1, x1], 3), name='Id')

			# recast as float for delta calculation
			x0 = tf.cast(x0, 'float32')
			x1 = tf.cast(x1, 'float32')
			y0 = tf.cast(y0, 'float32')
			y1 = tf.cast(y1, 'float32')

			# calculate deltas
			wa = (x1 - sampling_grid_x) * (y1 - sampling_grid_y)
			wb = (x1 - sampling_grid_x) * (sampling_grid_y - y0)
			wc = (sampling_grid_x - x0) * (y1 - sampling_grid_y)
			wd = (sampling_grid_x - x0) * (sampling_grid_y - y0)

			# add dimension for addition
			wa = tf.expand_dims(wa, axis=3)
			wb = tf.expand_dims(wb, axis=3)
			wc = tf.expand_dims(wc, axis=3)
			wd = tf.expand_dims(wd, axis=3)

			# compute output
			perturbed_image = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

			if data_format == 'NCHW':
				# convert back to NCHW to have consistency with the input
				perturbed_image = tf.transpose(perturbed_image, [0, 3, 1, 2])

			return perturbed_image


def flow_loss(flows, padding_mode='SYMMETRIC', epsilon=1e-8):
	"""Computes the flow loss designed to "enforce the locally smooth
	spatial transformation perturbation". See Eq. (4) in Xiao et al.
	(arXiv:1801.02612).

	Args:
		flows (tf.Tensor): flows of shape `(B, 2, H, W)`, where the second
						   dimension indicates the dimension on which the pixel
						   shift is applied.
		padding_mode (str): how to perform padding of the boundaries of the
							images. The value should be compatible with the
							`mode` argument of ``tf.pad``. Expected values are:
							* ``'SYMMETRIC'``: symmetric padding so as to not
							  penalize a significant flow at the boundary of
							  the images;
							* ``'CONSTANT'``: 0-padding of the boundaries so as
							  to enforce a small flow at the boundary of the
							  images.
		epsilon (float): small value added to the argument of ``tf.sqrt``
						 to prevent NaN gradients when the argument is zero.
	Returns:
		 1-D `tf.Tensor` of length `B` of the same type as `flows`.
	"""
	with tf.variable_scope('flow_loss'):
		# following the notation from Eq. (4):
		# \Delta u^{(p)} is flows[:, 1],
		# \Delta v^{(p)} is flows[:, 0], and
		# \Delta u^{(q)} is flows[:, 1] shifted by
		# (+1, +1), (+1, -1), (-1, +1), or (-1, -1) pixels
		# and \Delta v^{(q)} is the same but for shifted flows[:, 0]

		paddings = tf.constant([[0, 0], [0, 0], [1, 1], [1, 1]])
		padded_flows = tf.pad(
			flows, paddings, padding_mode, constant_values=0,
			name='padded_flows'
		)

		shifted_flows = [
			padded_flows[:, :, 2:, 2:],  # bottom right
			padded_flows[:, :, 2:, :-2],  # bottom left
			padded_flows[:, :, :-2, 2:],  # top right
			padded_flows[:, :, :-2, :-2]  # top left
		]

		return tf.reduce_sum(
			tf.add_n(
				[
					tf.sqrt(
						# ||\Delta u^{(p)} - \Delta u^{(q)}||_2^2
						(flows[:, 1] - shifted_flow[:, 1]) ** 2 +
						# ||\Delta v^{(p)} - \Delta v^{(q)}||_2^2
						(flows[:, 0] - shifted_flow[:, 0]) ** 2 +
						epsilon  # for numerical stability
					)
					for shifted_flow in shifted_flows
				]
			), axis=[1, 2], name='L_flow'
		)


# build graph
with tf.device("/gpu:0"):
	# ------ define input data ------
	inputImage = tf.placeholder(tf.float32, shape=[opt.batchSize, opt.inH, opt.inW, 3])
	renderTrans = tf.placeholder(tf.float32, shape=[opt.batchSize, opt.novelN, 4])
	depthGT = tf.placeholder(tf.float32, shape=[opt.batchSize, opt.novelN, opt.H, opt.W, 1])
	maskGT = tf.placeholder(tf.float32, shape=[opt.batchSize, opt.novelN, opt.H, opt.W, 1])
	flow = tf.placeholder(tf.float32, shape=[opt.batchSize, 2, opt.inH, opt.inW])
	spatialTransformedImage = flow_st(inputImage, flow)

	PH = [inputImage, renderTrans, depthGT, maskGT]
	# ------ build encoder-decoder ------
	encoder = graph.encoder if opt.arch == "original" else \
		graph.encoder_resnet if opt.arch == "resnet" else None
	decoder = graph.decoder if opt.arch == "original" else \
		graph.decoder_resnet if opt.arch == "resnet" else None
	latent = encoder(opt, spatialTransformedImage)
	# latent = encoder(opt, inputImage)
	XYZ, maskLogit = decoder(opt, latent)  # [B,H,W,3V],[B,H,W,V]
	mask = tf.to_float(maskLogit > 0)
	# ------ build transformer ------
	fuseTrans = tf.nn.l2_normalize(opt.fuseTrans, dim=1)
	XYZid, ML = transform.fuse3D(opt, XYZ, maskLogit, fuseTrans)  # [B,1,VHW]

	newDepth, newMaskLogit, collision = transform.render2D(opt, XYZid, ML, renderTrans)  # [B,N,H,W,1]

	# ------ define loss ------
	loss_depth = graph.masked_l1_loss(newDepth - depthGT, tf.equal(collision, 1)) / (opt.batchSize * opt.novelN)
	loss_mask = graph.cross_entropy_loss(newMaskLogit, maskGT) / (opt.batchSize * opt.novelN)
	loss_flow = tf.reduce_mean(flow_loss(flow))

	loss = loss_mask + opt.lambdaDepth * loss_depth + tau * loss_flow

	grad_inp = tf.gradients(loss_mask, inputImage)[0] + opt.lambdaDepth * tf.gradients(loss_depth, inputImage)[0]

	grad_flow = tf.gradients(loss_mask, flow)[0] \
								+ opt.lambdaDepth * tf.gradients(loss_depth, flow)[0] \
								+ tau * tf.gradients(loss_flow, flow)[0]

	# grad_inp = tf.clip_by_norm(tf.gradients(loss_mask, inputImage)[0] + opt.lambdaDepth * tf.gradients(loss_depth, inputImage)[0], clip_norm=1.0)
	#
	# grad_flow = tf.clip_by_norm(tf.gradients(loss_mask, flow)[0] \
	# 							+ opt.lambdaDepth * tf.gradients(loss_depth, flow)[0] \
	# 							+ tau * tf.gradients(loss_flow, flow)[0], clip_norm=1.0)

	# grad_inp = tf.clip_by_norm(tf.gradients(loss_mask, inputImage)[0], clip_norm=1.0) \
	# 		+ opt.lambdaDepth * tf.clip_by_norm(tf.gradients(loss_depth, inputImage)[0], clip_norm=1.0) \

	# grad_flow = tf.clip_by_norm(tf.gradients(loss_mask, flow)[0], clip_norm=1.0) \
	# 		   + opt.lambdaDepth * tf.clip_by_norm(tf.gradients(loss_depth, flow)[0], clip_norm=1.0) \
	# 		   + tau * tf.clip_by_norm(tf.gradients(loss_flow, flow)[0], clip_norm=1.0)

	# grad_inp = tf.gradients(loss, inputImage)[0]
	# grad_flow = tf.gradients(loss, flow)[0]

# load data
print(util.toMagenta("loading dataset..."))
dataloader = data.Loader(opt, loadTest=True)
CADN = len(dataloader.CADs)
chunkN = int(np.ceil(CADN / opt.chunkSize))
dataloader.loadChunk(opt, loadRange=[0, opt.chunkSize])

# prepare model saver/summary writer
saver = tf.train.Saver()

print(util.toYellow("======= EVALUATION START ======="))
timeStart = time.time()
# start session
tfConfig = tf.ConfigProto(allow_soft_placement=True)
tfConfig.gpu_options.allow_growth = True

source_img_path = 'source_image.npy'
# target_img_path = 'target_image.npy'
# target_renderTrans_path = 'target_renderTrans.npy'
# target_depthGT_path = 'target_depthGT.npy'
# target_maskGT_path = 'target_maskGT.npy'

target_counter = 0
def attack(sess, target_img, target_renderTrans, target_depthGT, target_maskGT, target_counter):
	print("Target Counter: %d"%target_counter)
	source_img = np.load(source_img_path)
	source_img = np.tile(source_img, (opt.batchSize, 1, 1, 1))

	runList = [XYZid, ML, loss, loss_depth, loss_mask, loss_flow, grad_inp, grad_flow]
	zero_flow = np.zeros((opt.batchSize, 2, opt.inH, opt.inW), dtype=np.object)

	target_batch = {inputImage: target_img, renderTrans: target_renderTrans, depthGT: target_depthGT,
					maskGT: target_maskGT, flow: zero_flow}

	xyz, ml, l, ld, lm, lf, l_grad, l_flow = sess.run(runList, feed_dict=target_batch)
	target_points = np.zeros([opt.batchSize, 1], dtype=np.object)
	for a in range(opt.batchSize):
		xyz1 = xyz[a].T  # [VHW,3]
		ml1 = ml[a].reshape([-1])  # [VHW]
		target_points[a, 0] = xyz1[ml1 > 0]

	x_adv = source_img.copy()
	flow_adv = zero_flow.copy()
	grad_inp_t = np.zeros(x_adv.shape, dtype=np.object)
	grad_flow_t = np.zeros(zero_flow.shape, dtype=np.object)

	pgd_max = source_img + attack_epsilon
	pgd_min = source_img - attack_epsilon

	alpha_inp = opt.alpha_inp
	alpha_flow = opt.alpha_flow
	iter_ = 0
	max_iters = 10000
	mu = 0.85

	while iter_ < max_iters:
		adv_batch = {inputImage: x_adv, renderTrans: target_renderTrans, depthGT: target_depthGT,
					 maskGT: target_maskGT, flow: flow_adv}
		xyz, ml, l, ld, lm, lf, l_grad, l_flow_grad = sess.run(runList, feed_dict=adv_batch)
		Vpred = np.zeros([opt.batchSize, 1], dtype=np.object)
		pred2GT_values = []
		GT2pred_values = []
		for a in range(opt.batchSize):
			xyz1 = xyz[a].T  # [VHW,3]
			ml1 = ml[a].reshape([-1])  # [VHW]
			Vpred[a, 0] = xyz1[ml1 > 0]
			pred2GT = computeTestError(Vpred[0][0], target_points[0][0], type="pred->GT") * 100
			GT2pred = computeTestError(target_points[0][0], Vpred[0][0], type="GT->pred") * 100
			pred2GT_values.append(pred2GT)
			GT2pred_values.append(GT2pred)
		# print(iter_, l, lm, ld, lf, "pred2GT:", pred2GT, "GT2pred:", GT2pred, flush=True)
		l2_loss = np.sum(np.power(x_adv - source_img,2))
		linf_loss = np.max(abs(x_adv-source_img))
		print(iter_, l, lm, ld, lf, linf_loss, "pred2GT:", sum(pred2GT_values)/opt.batchSize, "GT2pred:", sum(GT2pred_values)/opt.batchSize, flush=True)
		if iter_ == 0:
			grad_inp_t = l_grad/LA.norm(l_grad)
			grad_flow_t = l_flow_grad/LA.norm(l_flow_grad)
		else:
			grad_inp_t = mu * grad_inp_t + (1 - mu) * l_grad/LA.norm(l_grad)
			grad_flow_t = mu * grad_flow_t + (1 - mu) * l_flow_grad/LA.norm(l_flow_grad)

		if opt.attack_type == 'spatial_dag':
			if iter_ % 2 == 0:
				flow_adv = flow_adv - alpha_flow * grad_flow_t
			else:
				x_adv = np.clip(x_adv - alpha_inp * grad_inp_t, 0.0, 1.0)
				x_adv = np.clip(x_adv - pgd_max, None, 0) + pgd_max
				x_adv = np.clip(x_adv - pgd_min, 0, None) + pgd_min
		elif opt.attack_type == 'dag':
			x_adv = np.clip(x_adv - alpha_inp * grad_inp_t, 0.0, 1.0)
			x_adv = np.clip(x_adv - pgd_max, None, 0) + pgd_max
			x_adv = np.clip(x_adv - pgd_min, 0, None) + pgd_min
		elif opt.attack_type == 'spatial':
			flow_adv = flow_adv - alpha_flow * grad_flow_t
		iter_ += 1

		if iter_ % 1000 == 499:
			alpha_inp *= 0.75
			alpha_flow *= 0.75
		# tau *= 0.8
		#if iter_ % 500 == 499:

	adv_img = sess.run(spatialTransformedImage, feed_dict={inputImage: x_adv, flow: flow_adv})
	xyz, ml, _, _, _, _, _, _ = sess.run(runList, feed_dict= {inputImage: x_adv, renderTrans: target_renderTrans, depthGT: target_depthGT,
				 maskGT: target_maskGT, flow: flow_adv})
	Vpred = np.zeros([opt.batchSize, 1], dtype=np.object)
	for a in range(opt.batchSize):
		xyz1 = xyz[a].T  # [VHW,3]
		ml1 = ml[a].reshape([-1])  # [VHW]
		Vpred[a, 0] = xyz1[ml1 > 0]

	for image_index in range(adv_img.shape[0]):
		pred2GT = computeTestError(Vpred[image_index][0], target_points[image_index][0], type="pred->GT") * 100
		GT2pred = computeTestError(target_points[image_index][0], Vpred[image_index][0], type="GT->pred") * 100

		folder_name = '%s_%d_%d'%(opt.attack_type , target_counter, image_index)
		if not os.path.isdir('%s/%s'%(opt.save_dir, folder_name)):
			os.makedirs('%s/%s'%(opt.save_dir, folder_name))

		np.save('%s/%s/adv.npy' % (opt.save_dir, folder_name), adv_img[image_index])
		imageio.imwrite('%s/%s/adv_image.png' % (opt.save_dir, folder_name), (adv_img[image_index]*255).astype(np.uint8))
		np.save('%s/%s/points.npy' % (opt.save_dir, folder_name), Vpred[image_index][0])
		imageio.imwrite('%s/%s/target_image.png' % (opt.save_dir, folder_name), (target_img[image_index]*255).astype(np.uint8))

		with open("%s/%s/results.txt" % (opt.save_dir, folder_name) , "w") as file1:
			# Writing data to a file
			file1.write("pred2GT : %f \n GT2pred : %f \n " %(pred2GT, GT2pred))

with tf.Session(config=tfConfig) as sess:
	util.restoreModel(opt, sess, saver)
	print(util.toMagenta("loading pretrained ({0})...".format(opt.load)))

	target_img = np.load('target_img_path.npy')
	target_renderTrans = np.load('target_renderTrans_path.npy')
	target_depthGT = np.load('target_depthGT_path.npy')
	target_maskGT = np.load('target_maskGT_path.npy')
	attack(sess, target_img, target_renderTrans, target_depthGT, target_maskGT, target_counter)
	# for c in range(chunkN):
	# 	dataloader.shipChunk()
	# 	idx = np.arange(c*opt.chunkSize,min((c+1)*opt.chunkSize,CADN))
	# 	#if c!=chunkN-1:
	# 	dataloader.thread = threading.Thread(target=dataloader.loadChunk,
	# 										 args=[opt,[(c+1)*opt.chunkSize,min((c+2)*opt.chunkSize,CADN)]])
	# 	dataloader.thread.start()
	#
	# 	dataChunk = dataloader.readyChunk
	# 	modelIdx = np.random.permutation(opt.chunkSize)[:opt.batchSize]
	# 	modelIdxTile = np.tile(modelIdx, [opt.novelN, 1]).T
	# 	angleIdx = np.random.randint(24, size=[opt.batchSize])
	# 	sampleIdx = np.random.randint(opt.sampleN, size=[opt.batchSize, opt.novelN])
	# 	target_counter += 1
	# 	if dataChunk is not None:
	# 		target_img = dataChunk["image_in"][modelIdx, angleIdx]
	# 		target_renderTrans = dataChunk["trans"][modelIdxTile, sampleIdx]
	# 		target_depthGT = np.expand_dims(dataChunk["depth"][modelIdxTile, sampleIdx], axis=-1)
	# 		target_maskGT = np.expand_dims(dataChunk["mask"][modelIdxTile, sampleIdx], axis=-1)
	# 		opt.attack_type = 'spatial_dag'
	# 		attack(sess, target_img, target_renderTrans, target_depthGT, target_maskGT, target_counter)
	# 		opt.attack_type = 'dag'
	# 		attack(sess, target_img, target_renderTrans, target_depthGT, target_maskGT, target_counter)
	# 		opt.attack_type = 'spatial'
	# 		attack(sess, target_img, target_renderTrans, target_depthGT, target_maskGT, target_counter)

print(util.toYellow("======= EVALUATION DONE ======="))

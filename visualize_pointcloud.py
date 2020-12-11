import numpy as np
p1=np.load('points24_2.npy')
import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(p1)
o3d.visualization.draw_geometries([pcd])

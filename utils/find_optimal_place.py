import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
import sys
sys.path.insert(0, '..')

from utils.o3d_funcs import pcl_voxel
area = [-12.25, 12.25, -10.5, 10.5]

def perimeter_points(minx, maxx, miny, maxy):
    x = np.linspace(minx, maxx, 30)
    y = np.linspace(miny, maxy, 30)
    xv, yv = np.meshgrid(x, y)
    print(xv.max(),yv.max())
    points = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1)])
    points = points[(points[:,0] < 0) & (points[:,1] > 0) | (points[:,0] > 0) & (points[:,1] < 0)]
    return points

ref_p = np.fromfile('museum.bin', dtype='float32').reshape(-1, 4)[:, :3]    #perimeter_points(*area)
rot_mat = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_matrix()

ref_p = np.dot(ref_p, rot_mat.T)
ref_p = ref_p[np.linalg.norm(ref_p, axis=1) < 20]
ref_p = pcl_voxel(ref_p, voxel_size=0.1)[:, :2]

def objective(core_p, ref_p):
    core_p = core_p.reshape(-1, 2)
    return cdist(ref_p, core_p).min(axis=1).sum()

fig = plt.figure(figsize=(7, 6), dpi=700)
plt.rcParams.update({'font.size': 14})
initial_guess = np.zeros((1, 10))
result = minimize(lambda x: objective(x, ref_p), initial_guess, method='Powell')
core = result.x.reshape(-1, 2)
print(core)
plt.scatter(ref_p[:, 0], ref_p[:, 1], s=5, label='Point of interest')
plt.scatter(core[:, 0], core[:, 1], s=200, label='LiDARs position')
plt.legend()
fig.savefig('lidar_cust4.png', bbox_inches='tight')
plt.show()


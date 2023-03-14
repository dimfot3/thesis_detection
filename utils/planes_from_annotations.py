import pye57
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
from plane_detector import Plane, save_planes, readPlanes, visualize_planes, compute_convex_hull_on_plane, plot_plane_area

def get_pcl_plane_annot(pcl_path, annot_path):
    # Open E57 file
    annotations = pye57.E57(annot_path)
    pcl = np.fromfile(pcl_path, dtype='float32').reshape(-1, 4)[:, :3]
    # Get scan count
    scan_count = annotations.scan_count
    planes, inliers, normals, centers, coeff_arr= [], [], np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 4))
    tree = KDTree(pcl)
    # Loop through scans
    for i in range(scan_count):
        # Get scan data
        scan_data = annotations.read_scan_raw(i)
        # Do something with scan data
        # For example, get x, y, z coordinates
        x = np.array(scan_data['cartesianX']).reshape(-1, 1)
        y = np.array(scan_data['cartesianY']).reshape(-1, 1)
        z = np.array(scan_data['cartesianZ']).reshape(-1, 1)
        data = np.hstack([x, y, z])
        if(len(x)==4):
            v1 = data[1] - data[0]
            v2 = data[2] - data[0]
            normal = np.cross(v1, v2)
            normal /= np.linalg.norm(normal)
            normals = np.append(normals, np.array(normal).reshape(-1, 3), axis=0)
            centers = np.append(centers, np.array(data.mean(axis=0)).reshape(-1, 3), axis=0)
            coeff_arr = np.append(coeff_arr, np.array([*normal, -np.dot(centers[-1], normals[-1])]).reshape(-1, 4), axis=0)
        else:
            _, idxs = tree.query(data, k=1)
            inliers.append(list(idxs.reshape(-1, )))
    out_normals = []

    for i, plane_in in enumerate(inliers):
        a, b, c, d = coeff_arr[:, 0], coeff_arr[:, 1], coeff_arr[:, 2], coeff_arr[:, 3]
        distances = np.abs(np.dot(pcl[plane_in], [a, b, c]) + d)
        idxs = np.sum(distances, axis=0).argmin()
        planes.append(Plane(plane_in, normals[i]))
    return planes




# planes = get_pcl_plane_annot('../datasets/plane_detection_dataset/jrdb3.bin', '../datasets/jrdb3.e57')

# planes = readPlanes('../datasets/plane_detection_dataset/museum_ground_truth.bin')

# pcl = np.fromfile('../datasets/plane_detection_dataset/museum.bin', dtype='float32').reshape(-1, 4)[:, :3]
# areas = compute_convex_hull_on_plane(pcl, planes)
# # save_planes(planes, '../datasets/plane_detection_dataset/jrdb3_ground_truth.bin')
# visualize_planes(pcl, planes)
#plot_plane_area(pcl, planes, areas)



def create_cubic_pcl(width, height, depth, num_points, std_noise):
        # Generate random points on the surface of the cube
        points = []
        planes = [Plane(inliers=[], normal=[-1, 0, 0]), Plane(inliers=[], normal=[1, 0, 0]), Plane(inliers=[], normal=[0, -1, 0]), \
                  Plane(inliers=[], normal=[0, 1, 0]), Plane(inliers=[], normal=[0, 0, 1]), Plane(inliers=[], normal=[0, 0, -1])]
        for i in range(num_points):
            # Randomly choose a face of the cube
            face = np.random.choice(['left', 'right', 'bottom', 'top', 'front', 'back'])
            # Generate a random point on the chosen face
            if face == 'left':
                idx = 0
                point = np.array([-width/2, np.random.uniform(-height/2, height/2), np.random.uniform(-depth/2, depth/2)])
            elif face == 'right':
                idx = 1
                point = np.array([width/2, np.random.uniform(-height/2, height/2), np.random.uniform(-depth/2, depth/2)])
            elif face == 'bottom':
                idx = 2
                point = np.array([np.random.uniform(-width/2, width/2), -height/2, np.random.uniform(-depth/2, depth/2)])
            elif face == 'top':
                idx = 3
                point = np.array([np.random.uniform(-width/2, width/2), height/2, np.random.uniform(-depth/2, depth/2)])
            elif face == 'front':
                idx = 4
                point = np.array([np.random.uniform(-width/2, width/2), np.random.uniform(-height/2, height/2), depth/2])
            else:
                idx = 5
                point = np.array([np.random.uniform(-width/2, width/2), np.random.uniform(-height/2, height/2), -depth/2])
            points.append(point)
            planes[idx].inliers.append(i)
        # Convert the list of points to a NumPy array
        points = np.array(points) - std_noise + np.random.rand(num_points, 3) * std_noise/2
        points = np.append(points, np.zeros((points.shape[0], 1), dtype='float32'), axis=1)
        return points.astype('float32'), planes

points, planes = create_cubic_pcl(4, 4, 3, 8000, 0.3)
points.tofile('../datasets/plane_detection_dataset/box.bin')
visualize_planes(points, planes)
save_planes(planes, '../datasets/plane_detection_dataset/box_ground_truth.bin')
# def write_ply_file(points, filename):
#     """
#     Write a PLY file with x,y,z,intensity data.
#     """
#     with open(filename, 'w') as f:
#         # Write header
#         f.write('ply\n')
#         f.write('format ascii 1.0\n')
#         f.write('element vertex {}\n'.format(len(points)))
#         f.write('property float x\n')
#         f.write('property float y\n')
#         f.write('property float z\n')
#         f.write('property float intensity\n')
#         f.write('end_header\n')
#         # Write points
#         for point in points:
#             f.write('{} {} {} {}\n'.format(point[0], point[1], point[2], point[3]))
# points, filename = np.fromfile('../datasets/plane_detection_dataset/bedroom64.bin', dtype='float32').reshape(-1, 4), '../datasets/plane_detection_dataset//bedroom64.ply'
# write_ply_file(points, filename)
import numpy as np
from scipy.spatial.transform import Rotation as rot_mat
import wandb
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import trimesh
from o3d_funcs import plot_frame_annotation_kitti_v2, numpy_to_o3d
import pandas as pd
import hdbscan

def load_pcl(pcl_path):
    """
    load_pcl is loading a .pcd file as open3d Pointcloud

    :param pcl_path: the path where the .pcd is saved
    :return: open3d Pointcloud
    """
    pcd = np.fromfile(pcl_path, dtype='float32').reshape(-1, 4)[:, :3]
    return pcd

def get_point_annotations_kitti(pcl, dflabels, points_min=300):
    """
    anotate_pcl is used for annotation of https://jrdb.erc.monash.edu/. The
    annotations are in kitti format.
    It returns the points inside the 3d boxes of annotated objects

    :param dflabels: the labels of annotated pointclouds
    :return: the 3d boxes as list of open3d.geometry.OrientedBoundingBox
    """
    annotations = np.zeros(pcl.shape[0], dtype='bool')
    boxes = []
    for i in range(dflabels.shape[0]):
        if(dflabels['num_points'][i] < points_min):
            continue
        center = np.array([dflabels['cx'][i], dflabels['cy'][i], dflabels['cz'][i]])
        size = np.array([dflabels['w'][i], dflabels['l'][i], dflabels['h'][i]])
        trans_mat = np.zeros((4, 4))
        trans_mat[:3, :3], trans_mat[:3, 3], trans_mat[3, 3] = Rotation.from_euler('xyz', [0, 0, -dflabels['rot_z'][i]]\
                                                                , degrees=False).as_matrix(), center.reshape(-1, ), 1
        box_mesh = trimesh.creation.box(extents=size, transform=trans_mat)
        point_indices_inside_box = box_mesh.contains(pcl)
        annotations = annotations | point_indices_inside_box
        boxes.append(box_mesh)
    #plot_frame_annotation_kitti_v2(pcl, annotations)
    return annotations, boxes

def canonicalize_boxes(pcl, boxes, k, move_center=False):
    # normalize its origins
    centers = []
    pcl_tree = KDTree(pcl)
    for i, box in enumerate(boxes):
        centers.append(np.mean(box, axis=0))
    for i, box in enumerate(boxes):
        if (len(box) > 0) and (len(box) < k):
            dists, idxs = pcl_tree.query(box, k=2)
            dists, idxs = dists[:, 1], idxs[:, 1]
            rand_idxs = np.random.choice(idxs, k)
            boxes[i] = np.append(boxes[i], pcl[rand_idxs], axis=0)  - centers[i] * move_center
        elif len(box) > k:
            rand_idxs = np.random.choice(box.shape[0], k)
            boxes[i] = box[rand_idxs] - centers[i] * move_center
    return boxes, centers

def split_3d_point_cloud_overlapping(pcd, annotations, box_size, overlap_pt, pcl_box_num=2048, move_center=False, min_num_per_box=100):
    """
    Splits a 3D point cloud into overlapping boxes of a given size.
    :param pcd: numpy array of shape (N,3) containing the 3D point cloud
    :param annotations: mask array where True means point belongs to human
    :param box_size: the size of the boxes to split the point cloud into
    :param overlap_pt: the overlap between adjacent boxes as percentage (0, 1)
    :return: a list of tuples, each tuple containing a numpy array of shape (M,3) representing the points in the box,
             and a tuple of the box center coordinates. The points in the box are expressed in its center coordinates.
    """
    # Calculate the range of the point cloud in each dimension
    range_x = np.ptp(pcd[:, 0])
    range_y = np.ptp(pcd[:, 1])
    range_z = np.ptp(pcd[:, 2])
    overlap = overlap_pt * box_size
    # Calculate the number of boxes needed in each dimension
    num_boxes_x = int(np.ceil((range_x - box_size) / (box_size - overlap))) + 1
    num_boxes_y = int(np.ceil((range_y - box_size) / (box_size - overlap))) + 1
    num_boxes_z = int(np.ceil((range_z - box_size) / (box_size - overlap))) + 1
    # Initialize list of boxes
    boxes = []
    annotations_arr = []
    # Loop over all boxes
    for i in range(num_boxes_x):
        for j in range(num_boxes_y):
            for k in range(num_boxes_z):
                # Calculate the box center coordinates
                center_x = np.min(pcd[:, 0]) + (box_size - overlap) * i + box_size / 2
                center_y = np.min(pcd[:, 1]) + (box_size - overlap) * j + box_size / 2
                center_z = np.min(pcd[:, 2]) + (box_size - overlap) * k + box_size / 2
                # Get the points inside the box
                mask = ((pcd[:, 0] >= center_x - box_size / 2) & (pcd[:, 0] < center_x + box_size / 2)
                        & (pcd[:, 1] >= center_y - box_size / 2) & (pcd[:, 1] < center_y + box_size / 2)
                        & (pcd[:, 2] >= center_z - box_size / 2) & (pcd[:, 2] < center_z + box_size / 2))
                points_in_box = pcd[mask]
                annotations_in_box = annotations[mask]
                # Add the box to the list if it contains any points
                if (points_in_box.shape[0] > min_num_per_box) and ((annotations_in_box==True).sum() > 0):
                    boxes.append(points_in_box)
                    print(len(points_in_box))
    boxes, centers = canonicalize_boxes(pcd, boxes, pcl_box_num, move_center)
    return boxes, annotations_arr, centers

# def plot_frame_annotation_kitti_v2(pcl, annotations):
#     colors = ['red' if annot else 'blue' for annot in annotations]
#     fig = Figure()
#     canvas = FigureCanvas(fig)
#     ax = fig.gca()
#     ax.scatter(pcl[:, 0], pcl[:, 1], c=colors)
#     ax.axis('off')
#     fig.tight_layout(pad=0)
#     ax.margins(0)
#     fig.canvas.draw()
#     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     return wandb.Image(image)

def pcl_voxel(pcd, voxel_size=0.1):
    min_coords = np.min(pcd, axis=0)
    max_coords = np.max(pcd, axis=0)
    n_voxels = np.ceil((max_coords - min_coords) / voxel_size).astype(int)
    voxel_indices = np.floor((pcd - min_coords) / voxel_size).astype(int)
    unique_indices = np.unique(voxel_indices, axis=0)
    tree = KDTree(pcd)
    centers = (unique_indices + 0.5) * voxel_size + min_coords
    _, indices = tree.query(centers)
    return pcd[indices, :]


def split_point_cloud(points, K):
    # Cluster the points using HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=30, prediction_data=True).fit(points)
    labels = clusterer.labels_
    n_clusters = max(labels) + 1
    
    # Find the size of each cluster
    cluster_sizes = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_sizes[i] = np.sum(labels == i)

    # Sort the clusters by size (descending order)
    sorted_clusters = np.argsort(cluster_sizes)[::-1]
    # Initialize the new point cloud
    new_points = np.empty((0, 3))
    membeship = hdbscan.membership_vector(clusterer, points[:1000])
    print(membeship.shape)
    exit()
    # Iterate over the clusters, adding points until K is reached
    for i in sorted_clusters:
        cluster_points = points[labels == i]
        n_points = cluster_points.shape[0]

        # If the cluster has fewer than K points, add all of them
        if n_points <= K:
            new_points = np.concatenate((new_points, cluster_points))
            # test_points = points[~(labels == i)]
            # cluster_density = hdbscan.probability_density(cluster_points, single_linkage_tree)
            # cluster_density.argsort()[::-1][0:K-n_points]
        # If the cluster has more than K points, randomly subsample to get K points
        else:
            subsample_indices = np.random.choice(n_points, size=K, replace=False)
            subsample_points = cluster_points[subsample_indices, :]
            new_points = np.concatenate((new_points, subsample_points))

    return new_points[:K*n_clusters]






if __name__ == '__main__':
    pcl = load_pcl('../datasets/JRDB/velodyne/000000.bin')
    pcl = pcl[np.linalg.norm(pcl, axis=1) < 15]
    label_cols = ['obs_angle', 'l', 'w', 'h', 'cx', 'cy', 'cz', 'rot_z', 'num_points']
    labels = pd.read_csv('../datasets/JRDB/labels/000000.txt', sep=' ', header=None, names=label_cols)
    pcl_voxeled = pcl_voxel(pcl, voxel_size=0.1)
    arr = split_point_cloud(pcl_voxeled, 2048)
    # for label in np.unique(labels):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     ax.scatter(pcl_voxeled[labels == label, 0], pcl_voxeled[labels == label, 1], pcl_voxeled[labels == label, 2], label=f"Class {label}")
    #     plt.show()
    # plt.hist(labels)
    # plt.show()
    

    # annotations, boxes = get_point_annotations_kitti(pcl_voxeled, labels, points_min=200)
    # split_3d_point_cloud_overlapping(pcl_voxeled, annotations, 7, 0.4, pcl_box_num=4096, move_center=False, min_num_per_box=300)
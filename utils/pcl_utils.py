import numpy as np
from scipy.spatial.transform import Rotation as rot_mat
import wandb
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import trimesh
#from o3d_funcs import plot_frame_annotation_kitti_v2, numpy_to_o3d
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
        # if(dflabels['num_points'][i] < points_min):
        #     continue
        center = np.array([dflabels['cx'][i], dflabels['cy'][i], dflabels['cz'][i]])
        size = np.array([dflabels['w'][i], dflabels['l'][i], dflabels['h'][i]]) * 1.05
        trans_mat = np.zeros((4, 4))
        trans_mat[:3, :3], trans_mat[:3, 3], trans_mat[3, 3] = Rotation.from_euler('xyz', [0, 0, dflabels['rot_z'][i]]\
                                                                , degrees=False).as_matrix(), center.reshape(-1, ), 1
        box_mesh = trimesh.creation.box(extents=size, transform=trans_mat)
        point_indices_inside_box = box_mesh.contains(pcl)
        annotations = annotations | point_indices_inside_box
        boxes.append(np.argwhere(point_indices_inside_box).reshape(-1, ))
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

def plot_frame_annotation_kitti_v2(pcl, annotations):
    colors = ['red' if annot else 'blue' for annot in annotations]
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.scatter(pcl[:, 0], pcl[:, 1], c=colors)
    ax.axis('off')
    fig.tight_layout(pad=0)
    ax.margins(0)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return wandb.Image(image)

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

def check_cluster_valdity(cluster_points, core_indices, min_cluster_size=30, max_size=2):
    tree = KDTree(cluster_points[core_indices])
    dists, _ = tree.query(cluster_points[core_indices], k=2)
    dists = dists[:, 1]
    if (dists < max_size).sum() > min_cluster_size:
        return True
    return False

def split_point_cloud_adaptive_training(points, annot_labels, K, min_cluster_size=30, max_size_core=2, move_center=True, points_min=100):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True, cluster_selection_epsilon=0.2).fit(points)
    labels = clusterer.labels_
    n_clusters = max(labels)
    cluster_arr_idxs = []
    tree = KDTree(points)
    for i in range(n_clusters):
        cluster_idxs = np.argwhere(labels == i).reshape(-1, )
        cluster_points = points[cluster_idxs]
        n_points = cluster_points.shape[0]
        core_indices = clusterer.outlier_scores_[cluster_idxs] < 0.00001
        test_bool = check_cluster_valdity(cluster_points, core_indices, min_cluster_size, max_size=max_size_core)       # filter out higly sparse clusters 
        if not test_bool: continue
        if n_points <= K:       # deal with less than K points
            _, idxs = tree.query([np.median(cluster_points, axis=0)], k=K)
            new_idxs = np.setdiff1d(idxs.reshape(-1, ), cluster_idxs)[:K-n_points]
            cluster_idxs = np.concatenate((cluster_idxs, new_idxs))
        else:                   # deal with more than K points
            subsample_indices = np.random.choice(n_points, size=K, replace=False)
            cluster_idxs = cluster_idxs[subsample_indices]
        cluster_arr_idxs.append(cluster_idxs)
    # out_pcls = [points[idxs] for idxs in cluster_arr_idxs]
    _, boxes = get_point_annotations_kitti(points, annot_labels, points_min=points_min)
    annotations = []
    out_pcls = []
    for cluster_idxs in cluster_arr_idxs:
        curr_annot = np.zeros((cluster_idxs.shape[0], ), dtype='float32')
        out_pcls.append(points[cluster_idxs])
        for box in boxes:
            com_idxs = np.intersect1d(cluster_idxs, box, return_indices=True)
            if com_idxs[0].shape[0] == 0: continue
            curr_annot[com_idxs[1]] = (com_idxs[0].shape[0] / len(box)) * (np.min([len(box) / points_min, 1]))
        annotations.append(curr_annot)
    centers = np.zeros((len(out_pcls), 3))
    if move_center: 
        out_pcls = [cluster  - np.mean(cluster, axis=0) for cluster in out_pcls]
        centers = np.array([np.mean(cluster, axis=0) for cluster in out_pcls])
    return out_pcls, annotations, centers


if __name__ == '__main__':
    pcl = load_pcl('../datasets/JRDB/velodyne/000003.bin')
    #pcl = pcl[np.linalg.norm(pcl, axis=1) < 15]
    label_cols = ['obs_angle', 'l', 'w', 'h', 'cx', 'cy', 'cz', 'rot_z', 'num_points']
    labels = pd.read_csv('../datasets/JRDB/labels/000003.txt', sep=' ', header=None, names=label_cols)
    pcl_voxeled = pcl_voxel(pcl, voxel_size=0.1)
    out_pcls, annotations, centers = split_point_cloud_adaptive_training(pcl_voxeled, labels, 2048,
                                                         min_cluster_size=30, max_size_core=1, move_center=True, points_min=100)
    for i, (curr_pcl, curr_annot) in enumerate(zip(out_pcls, annotations)):
        colors = np.zeros((curr_annot.shape[0], 3))
        colors[:, 0] = curr_annot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(curr_pcl[:, 0], curr_pcl[:, 1], curr_pcl[:, 2], c=colors)
        plt.show()

import numpy as np
from scipy.spatial.transform import Rotation as rot_mat
import wandb
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


def load_pcl(pcl_path):
    """
    load_pcl is loading a .pcd file as open3d Pointcloud

    :param pcl_path: the path where the .pcd is saved
    :return: open3d Pointcloud
    """
    pcd = np.fromfile(pcl_path, dtype='float32').reshape(-1, 4)[:, :3]
    return pcd

def box_vertices_indices(center, rotationz, width, height, depth):
    corners_local = np.array([
        [-0.5*width, -0.5*height, -0.5*depth],
        [-0.5*width,  0.5*height, -0.5*depth],
        [ 0.5*width,  0.5*height, -0.5*depth],
        [ 0.5*width, -0.5*height, -0.5*depth],
        [-0.5*width, -0.5*height,  0.5*depth],
        [-0.5*width,  0.5*height,  0.5*depth],
        [ 0.5*width,  0.5*height,  0.5*depth],
        [ 0.5*width, -0.5*height,  0.5*depth]
    ])
    rotation_matrix = Rotation.from_euler('xyz', [0, 0, rotationz], degrees=False).as_matrix()
    corners_rotated = np.dot(corners_local, rotation_matrix.T)
    corners_translated = corners_rotated + center.reshape(-1, 3)
    vertex_indices_3d = np.zeros((8, 3), dtype=float)
    for i in range(8):
        vertex_indices_3d[i] = corners_translated[i]
    return vertex_indices_3d

def get_point_annotations_kitti(pcl, dflabels, points_min=300):
    """
    anotate_pcl is used for annotation of https://jrdb.erc.monash.edu/. The
    annotations are in kitti format.
    It returns the points inside the 3d boxes of annotated objects

    :param dflabels: the labels of annotated pointclouds
    :return: the 3d boxes as list of open3d.geometry.OrientedBoundingBox
    """
    annotations = np.zeros(pcl.shape[0], dtype='bool')
    for i in range(dflabels.shape[0]):
        if(dflabels['num_points'][i] < points_min):
            continue
        center = np.array([dflabels['cx'][i], dflabels['cy'][i], dflabels['cz'][i]])
        size = np.array([dflabels['l'][i], dflabels['w'][i], dflabels['h'][i]])
        box3d = box_vertices_indices(center, -dflabels['rot_z'][i], size[0], size[1], size[2])
        minx, maxx = box3d[:, 0].min(), box3d[:, 0].max()
        miny, maxy = box3d[:, 1].min(), box3d[:, 1].max()
        minz, maxz = box3d[:, 2].min(), box3d[:, 2].max()
        annotations = annotations | ((pcl[:, 0] > minx) & (pcl[:, 0] < maxx) & (pcl[:, 1] > miny) & \
             (pcl[:, 1] < maxy) & (pcl[:, 2] > minz) & (pcl[:, 2] < maxz))
    return annotations

def canonicalize_boxes(boxes, annots, k, move_center=False):
    # normalize its origins
    centers = []
    for i, box in enumerate(boxes):
        centers.append(np.mean(box, axis=0))
    # normalize its dimension
    for i, (box, annot) in enumerate(zip(boxes, annots)):
        if(len(box) == 0):
            print('here')
        if (len(box) > 0) and (len(box) < k):
            randidxs = np.random.choice(box.shape[0], k - box.shape[0])
            boxes[i] = np.append(boxes[i], box[randidxs], axis=0)  - centers[i] * move_center
            annots[i] = np.append(annots[i], annot[randidxs])
        elif len(box) > k:
            rand_idxs = np.random.choice(box.shape[0], k)
            boxes[i] = box[rand_idxs] - centers[i] * move_center
            annots[i] = annot[rand_idxs]
    return boxes, annots, centers

def merge_boxes(boxes, annots, k):
    merged_boxes = []
    merged_annot = []
    current_box = np.array([]).reshape(-1, 3)
    current_annot = np.array([], dtype='bool')
    current_sum = 0
    for box, annot in zip(boxes, annots):
        if current_sum + len(box) <= k:
            current_box = np.append(current_box, box, axis=0)
            current_annot = np.append(current_annot, annot)
            current_sum += len(box)
        elif current_box.shape[0] > 0:
            merged_boxes.append(current_box)
            merged_annot.append(current_annot)
            current_box = box
            current_annot = annot
            current_sum = len(box)
    if current_box.shape[0] > 0:
        merged_boxes.append(current_box)
        merged_annot.append(current_annot)
    return merged_boxes, merged_annot

def split_3d_point_cloud_overlapping(pcd, annotations, box_size, overlap_pt, pcl_box_num=2048, move_center=False, min_num_per_box=300):
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
    annotations_splitted = []
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
                if (points_in_box.shape[0] > min_num_per_box) and ((annotations_in_box==True).sum() > min_num_per_box * 0.08):
                    boxes.append(points_in_box)
                    annotations_splitted.append(annotations_in_box)
    boxes, annotations_splitted = merge_boxes(boxes, annotations_splitted, pcl_box_num)
    boxes, annotations_splitted, centers = canonicalize_boxes(boxes, annotations_splitted, pcl_box_num, move_center)
    return boxes, annotations_splitted, centers

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
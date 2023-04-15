import numpy as np
import matplotlib.pyplot as plt
import cudf
import numba as nb
from cuml.cluster import HDBSCAN
import cuml
from scipy.spatial import KDTree
from time import time
import cupy as cp
import torch


def create_hclusters(pcl_gpu, min_cluster_size=50):
    # HDBSCAN
    pcl_gpu_df = cudf.DataFrame(pcl_gpu)
    hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', prediction_data=True)
    labels = hdbscan.fit_predict(pcl_gpu_df)
    memberships = cuml.cluster.hdbscan.all_points_membership_vectors(hdbscan)
    labels = labels.to_numpy()
    del pcl_gpu_df
    return labels, memberships

def merge_degenerared_clusters(labels, memberships, cluster_shape=2048):
    classes, cluster_num = np.unique(labels, return_counts=True)
    cluster_num, classes = cluster_num[classes>=0], classes[classes>=0] 
    sorted_idxs = np.argsort(cluster_num)
    classes, cluster_num = classes[sorted_idxs], cluster_num[sorted_idxs]
    while True:
        if cluster_num[0] >= cluster_shape/4:
            break
        cand_class = classes[1:][cluster_num[1:] < cluster_shape]
        if len(cand_class) == 0: break
        new_classes = np.argmax(memberships[labels==classes[0], :][:, cand_class], axis=1).reshape(-1, )
        labels[labels==classes[0]] = cand_class[new_classes]
        classes, cluster_num = np.unique(labels, return_counts=True)
        sorted_idxs = np.argsort(cluster_num)
        classes, cluster_num = classes[sorted_idxs], cluster_num[sorted_idxs]
    return labels, memberships

def canonicalize_cluster(pcl, labels, memberships, cluster_shape=2048):
    # canonicalize clusters
    cluster_idxs = []
    tree = KDTree(pcl)
    for class_idx in np.unique(labels):
        if class_idx == -1: continue
        curr_points = pcl[labels==class_idx]
        curr_idxs = np.argwhere(labels==class_idx).reshape(-1, )
        if curr_points.shape[0] == cluster_shape:
            cluster_idxs.append(curr_idxs)
        elif curr_points.shape[0] < cluster_shape:
            _, idxs = tree.query([curr_points.mean(axis=0)], k=cluster_shape)
            new_idxs = np.setdiff1d(idxs.reshape(-1, ), curr_idxs)[:cluster_shape-curr_points.shape[0]]        
            cluster_idxs.append(np.concatenate([np.argwhere(labels==class_idx).reshape(-1, ), new_idxs], axis=0))
        else:
            cluster_idxs.append(curr_idxs[[np.argsort(memberships[labels==class_idx, :][:, class_idx])[::-1][:cluster_shape]]])
    return cluster_idxs

def split_pcl_to_clusters(pcl, cluster_shape=2048, min_cluster_size=50, return_pcl_gpu=False):
    labels, memberships = create_hclusters(pcl, min_cluster_size)
    labels, memberships = merge_degenerared_clusters(labels, memberships, cluster_shape)
    cluster_idxs = canonicalize_cluster(pcl, labels, memberships, cluster_shape)
    if return_pcl_gpu:
        tensor_arr = []
        center_arr = []
        for idxs in cluster_idxs:
            center_arr.append(pcl[idxs].mean(axis=0))
            tensor_arr.append(torch.tensor(pcl[idxs]).reshape(2048, 3).to('cuda:0').type(torch.cuda.FloatTensor) \
                               - torch.Tensor(center_arr[-1]).type(torch.cuda.FloatTensor).to('cuda:0'))
        tensor_3d = torch.stack(tensor_arr) if len(tensor_arr) > 0 else None
        return cluster_idxs, tensor_3d, center_arr
    return cluster_idxs

if __name__ == '__main__':
    from o3d_funcs import pcl_voxel
    pcl = np.fromfile('test.bin', dtype='float32').reshape(-1, 3)
    pcl = pcl_voxel(pcl, voxel_size=0.14)
    cluster_arr = split_pcl_to_clusters(pcl, cluster_shape=2048)
    ax = plt.subplot(111, projection='3d')
    for cluster_idxs in cluster_arr:
        ax.scatter(pcl[cluster_idxs, 0], pcl[cluster_idxs, 1], pcl[cluster_idxs, 2])
    plt.show()

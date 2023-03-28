import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, '../utils/')
from pcl_utils import load_pcl, get_point_annotations_kitti, split_3d_point_cloud_overlapping, \
    plot_frame_annotation_kitti_v2, pcl_voxel

def get_file_names(data_path):
    pcl_files = [file.replace('.bin', '') for file in os.listdir(data_path + 'velodyne/')]
    label_files = [file.replace('.txt', '') for file in os.listdir(data_path + 'labels/')]
    labeled_pcls = np.intersect1d(pcl_files, label_files)
    pcl_files = [file + '.bin' for file in labeled_pcls]
    label_files = [file + '.txt' for file in labeled_pcls]
    return pcl_files, label_files

def save_pointclouds_to_hdpy(filename, data_path, compression='gzip', compression_level=6, preprocess_args={}):
    with h5py.File(filename, 'w') as f:
        label_cols = ['obs_angle', 'l', 'w', 'h', 'cx', 'cy', 'cz', 'rot_z', 'num_points']
        pcl_files, label_files = get_file_names(data_path)
        points_dset = f.create_dataset('pcls', shape=(1, preprocess_args['input_len'], 3), maxshape=(None, preprocess_args['input_len'], 3), \
                                       dtype=np.float32, compression=compression, compression_opts=compression_level)
        annotations_dset = f.create_dataset('annotations', shape=(1, preprocess_args['input_len']), maxshape=(None, preprocess_args['input_len']), \
                                       dtype=np.bool_, compression=compression, compression_opts=compression_level)
        center_dset = f.create_dataset('centers', shape=(1, 3), maxshape=(None, 3), \
                                       dtype=np.float32, compression=compression, compression_opts=compression_level)
        total_plcs = 0
        for idx, (pcl_file, label_file) in tqdm(enumerate(zip(pcl_files, label_files)), total=len(pcl_files)):
            pcl = load_pcl(data_path + 'velodyne/' + pcl_file)
            labels = pd.read_csv(data_path + 'labels/' + label_file, sep=' ', header=None, names=label_cols)
            pcl_voxeled = pcl_voxel(pcl, voxel_size=preprocess_args['voxel_down'])
            annotations = get_point_annotations_kitti(pcl_voxeled, labels, points_min=200)
            splitted_pcl, splitted_ann, centers = split_3d_point_cloud_overlapping(pcl_voxeled, annotations, box_size=preprocess_args['box_size'], \
            overlap_pt=preprocess_args['overlap_pt'], pcl_box_num=preprocess_args['input_len'], move_center=preprocess_args['move_center'], min_num_per_box=300)
            for subidx in range(len(splitted_pcl)):
                points_dset.resize((total_plcs + 1, preprocess_args['input_len'], 3))
                annotations_dset.resize((total_plcs + 1, preprocess_args['input_len']))
                center_dset.resize((total_plcs + 1, 3))
                points_dset[total_plcs, :, :] = splitted_pcl[subidx]
                annotations_dset[total_plcs, :] = splitted_ann[subidx]
                center_dset[total_plcs, :] = centers[subidx].reshape(1, 3)
                total_plcs += 1

            #annotations_dset[i, :num_points] = annotation

def load_pointclouds_from_hdpy(filename):
    with h5py.File(filename, 'r') as f:
        pcls = f['pcls'][:]
        annotations = f['annotations'][:]
        centers = f['centers'][:]
    return pcls, annotations, centers


preprocess_args = {'input_len': 4096, 
                    'voxel_down': 0.1, 
                    'move_center': True,
                    'box_size': 9,
                    'overlap_pt': 0.25
                    }

save_pointclouds_to_hdpy('./JRDB/test_data.h5py', './JRDB/', compression='gzip', compression_level=6, preprocess_args=preprocess_args)
pcls, annotations, centers = load_pointclouds_from_hdpy('test.h5py')
print(annotations.shape, centers.shape)
for i in range(pcls.shape[0]):
    center =  centers[i].reshape(1, 3)
    pcl = pcls[i]
    annot = annotations[i]
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.scatter(pcl[~annot, 0], pcl[~annot, 1], pcl[~annot, 2])
    ax.scatter(pcl[annot, 0], pcl[annot, 1], pcl[annot, 2])
    plt.show()
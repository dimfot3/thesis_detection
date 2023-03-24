import torch
import os, sys
sys.path.insert(0, '../')
from torch.utils.data import Dataset, DataLoader
from pcl_utils import load_pcl, get_point_annotations_kitti, split_3d_point_cloud_overlapping, \
    plot_frame_annotation_kitti_v2, pcl_voxel
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class humanDBLoader(Dataset):
    def __init__(self, data_path, pcl_len=2048, move_center=False, voxel=0.1):
        pcl_files = [file.replace('.bin', '') for file in os.listdir(data_path + 'velodyne/')]
        label_files = [file.replace('.txt', '') for file in os.listdir(data_path + 'labels/')]
        labeled_pcls = np.intersect1d(pcl_files, label_files)
        self.data_path = data_path
        self.pcl_files = [file + '.bin' for file in labeled_pcls]
        self.label_files = [file + '.txt' for file in labeled_pcls]
        self.label_cols = ['obs_angle', 'l', 'w', 'h', 'cx', 'cy', 'cz', 'rot_z', 'num_points']
        self.pcl_len = pcl_len
        self.move_center = move_center
        self.box_size = 5
        self.overlap_pt = 0.4
        self.voxel = voxel

    def __len__(self):
        return len(self.pcl_files)

    def __getitem__(self, idx):
        # load pcl and its annotations
        pcl = load_pcl(self.data_path + 'velodyne/' + self.pcl_files[idx])
        labels = pd.read_csv(self.data_path + 'labels/' + self.label_files[idx], sep=' ', header=None, names=self.label_cols)
        # transform the pcl and annotations
        # voxel downsample
        pcl_voxeled = pcl_voxel(pcl, voxel_size=self.voxel)
        annotations = get_point_annotations_kitti(pcl_voxeled, labels, points_min=200)
        pcl_numpy = pcl_voxeled
        # 3d tiling
        splitted_pcl, splitted_ann, centers = split_3d_point_cloud_overlapping(pcl_numpy, annotations, box_size=self.box_size, \
            overlap_pt=self.overlap_pt, pcl_box_num=self.pcl_len, move_center=self.move_center, min_num_per_box=300)
        return splitted_pcl, splitted_ann, centers

def custom_collate(batch):
    # Stack the boxes from each sample into a single tensor
    pcls_tiled = []
    annot_tiled = []
    centers_tiled = []
    for pcls, annotations, centers in batch:
        pcls_tiled += pcls
        annot_tiled += annotations
        centers_tiled += centers
    batched_idxs = np.array_split(np.arange(len(pcls_tiled)), np.ceil(len(pcls_tiled) / len(batch)))
    pcls_batched = [pcls_tiled[idxs[0]:(idxs[-1]+1)] for idxs in batched_idxs]
    annot_batched = [annot_tiled[idxs[0]:(idxs[-1]+1)] for idxs in batched_idxs]
    centers_batched = [centers_tiled[idxs[0]:(idxs[-1]+1)] for idxs in batched_idxs]
    # transform to tensor
    for i, (pcl_arr, annot_arr) in enumerate(zip(pcls_batched, annot_batched)):
        batch_pcl_tensors = [torch.from_numpy(arr).unsqueeze(0).type(torch.float32) for arr in pcl_arr] 
        batch_annot_tensors = [torch.from_numpy(arr).unsqueeze(0).type(torch.float32) for arr in annot_arr]
        pcls_batched[i] = torch.cat(batch_pcl_tensors, dim=0)
        annot_batched[i] = torch.cat(batch_annot_tensors, dim=0)
    return pcls_batched, annot_batched, centers_batched

if __name__ == '__main__':
    path = '../datasets/JRDB/'
    dataset = humanDBLoader(path, pcl_len=2048, move_center=True, voxel=0.1)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=5, collate_fn=custom_collate)
    human = []
    no_human = []
    for pcls, annotations, centers in tqdm(dataloader, total=len(dataloader)):
        for i in range((len(pcls))):
            annot = annotations[i][0].detach().cpu().numpy().astype('bool')
            pcl = pcls[i][0].detach().cpu().numpy()
            ax = plt.subplot(1, 1, 1, projection='3d')
            ax.scatter(pcl[~annot, 0], pcl[~annot, 1], pcl[~annot, 2], c='blue')
            ax.scatter(pcl[annot, 0], pcl[annot, 1], pcl[annot, 2], c='red')
            plt.show()
    # import matplotlib.pyplot as plt
    # plt.hist(no_human, label='no human')
    # plt.hist(human, label='human')
    # plt.title('human')
    # plt.show()
import torch
import os, sys
sys.path.insert(0, '../')
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py

class humanDBLoader(Dataset):
    def __init__(self, data_path):
        self.data_file = h5py.File(data_path, 'r')

    def __len__(self):
        return self.data_file['pcls'].len()

    def __getitem__(self, idx):
        splitted_pcl, splitted_ann, centers = self.data_file['pcls'][idx], self.data_file['annotations'][idx], \
            self.data_file['centers'][idx]
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
    path = '../datasets/test.h5py'
    dataset = humanDBLoader(path)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=5)
    human = []
    no_human = []
    for pcls, annotations, centers in tqdm(dataloader, total=len(dataloader)):
        pcl, annot, center = pcls[0], annotations[0], centers[0], 
        
        ax = plt.subplot(1, 1, 1, projection='3d')
        ax.scatter(pcl[~annot, 0], pcl[~annot, 1], pcl[~annot, 2], c='blue')
        ax.scatter(pcl[annot, 0], pcl[annot, 1], pcl[annot, 2], c='red')
        plt.show()
    # import matplotlib.pyplot as plt
    # plt.hist(no_human, label='no human')
    # plt.hist(human, label='human')
    # plt.title('human')
    # plt.show()
import torch
import os, sys
sys.path.insert(0, '../')
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
from DataAugemnetations import Augmentator

class humanDBLoader(Dataset):
    def __init__(self, data_path, batch_size=32, augmentation=False):
        self.data_file = h5py.File(data_path, 'r')
        self.data_len = self.data_file['pcls'].len()
        self.batch_size = batch_size
        self.batch_num = np.ceil(self.data_len // self.batch_size).astype('int')
        self.augmentation = augmentation
        self.augmentator = Augmentator(remove_p=0.3, noise_std=0.08, add_points_p=0.05, rot_prob=[0.1, 0.6, 0.2])
        
    def __len__(self):
        return self.batch_num

    def __getitem__(self, idx):
        start_idx, end_idx = idx * self.batch_size, min((idx + 1) * self.batch_size, self.batch_num * self.batch_size)
        splitted_pcl, splitted_ann, centers = self.data_file['pcls'][start_idx:end_idx], \
                                        self.data_file['annotations'][start_idx:end_idx], \
                                        self.data_file['centers'][start_idx:end_idx]
        if self.augmentation:
            splitted_pcl, splitted_ann = self.augmentator.apply_augmentation(splitted_pcl, splitted_ann)
        return splitted_pcl.astype('float32'), splitted_ann.astype('float32'), centers.astype('float32')


if __name__ == '__main__':
    path = '/media/visitor1/DBStorage/Datasets/JRDB/training_format/train_data.h5py'
    dataset = humanDBLoader(path, batch_size=16)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=True)
    human = []
    no_human = []
    for pcls, annotations, centers in tqdm(dataloader, total=len(dataloader)):
        pcl, annot, center = pcls[0], annotations[0], centers[0]
        # ax = plt.subplot(1, 1, 1, projection='3d')
        # ax.scatter(pcl[~annot, 0], pcl[~annot, 1], pcl[~annot, 2], c='blue')
        # ax.scatter(pcl[annot, 0], pcl[annot, 1], pcl[annot, 2], c='red')
        # plt.show()
    # import matplotlib.pyplot as plt
    # plt.hist(no_human, label='no human')
    # plt.hist(human, label='human')
    # plt.title('human')
    # plt.show()
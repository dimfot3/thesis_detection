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
    def __init__(self, data_path, batch_size=32):
        self.data_file = h5py.File(data_path, 'r')
        self.data_len = self.data_file['pcls'].len()
        self.batch_size = batch_size
        self.batch_num = np.ceil(self.data_len // self.batch_size).astype('int')

    def __len__(self):
        return self.batch_num

    def __getitem__(self, idx):
        start_idx, end_idx = idx * self.batch_size, min((idx + 1) * self.batch_size, self.batch_num * self.batch_size)
        splitted_pcl, splitted_ann, centers = self.data_file['pcls'][start_idx:end_idx], \
                                        self.data_file['annotations'][start_idx:end_idx], \
                                        self.data_file['centers'][start_idx:end_idx]
        return splitted_pcl, splitted_ann, centers


if __name__ == '__main__':
    path = '/media/visitor1/DBStorage/Datasets/JRDB/training_format/training_data.h5py'
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
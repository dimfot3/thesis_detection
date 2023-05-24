import numpy as np
import torch
import h5py
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

class Augmentator:
    def __init__(self, remove_p=0.01, noise_std=0.01, add_points_p=0.05, rot_prob=[0.1, 0.1, 0.2]):
        self.remove_p = remove_p
        self.noise_std = noise_std
        self.add_points_p = add_points_p
        self.rot_prob = rot_prob

    def remove_points(self, pcl, labels):
        B, N, _ = pcl.shape
        K = np.floor(self.remove_p * N * np.clip(np.random.rand(), 0, self.remove_p)).astype('int')
        keep_mask = np.ones((B, N), dtype=bool)
        for i in range(B):
            keep_mask[i, np.random.choice(N, size=K, replace=False)] = False
        pcl_augmented = []
        labels_augmented = []
        for i in range(B):
            pcl_augmented.append(pcl[i][keep_mask[i]])
            labels_augmented.append(labels[i][keep_mask[i]])
        pcl_augmented = np.stack(pcl_augmented, axis=0)
        labels_augmented = np.stack(labels_augmented, axis=0)
        return pcl_augmented, labels_augmented
    
    def add_noise(self, pcl):
        noise = - self.noise_std / 2 + np.random.random(pcl.shape) * self.noise_std / 2
        pcl_w_noise = pcl + noise
        return pcl_w_noise

    def add_random_points(self, pcl, labels):
        B, N, _ = pcl.shape
        K = np.floor(self.add_points_p * pcl.shape[1]).astype('int')
        points_to_add = np.random.random((B, K, 3)) - 0.5
        points_to_add[:, :, 0], points_to_add[:, :, 1], points_to_add[:, :, 2] = \
        pcl[:, :, 0].ptp(axis=1).reshape(-1, 1) * points_to_add[:, :, 0], pcl[:, :, 1].ptp(axis=1).reshape(-1, 1) * points_to_add[:, :, 1], \
        pcl[:, :, 2].ptp(axis=1).reshape(-1, 1) * points_to_add[:, :, 2]
        labels_to_add = np.zeros((B, K), dtype=bool)
        pcl = np.append(pcl, points_to_add, axis=1)
        labels = np.append(labels, labels_to_add, axis=1)
        shuffle_idxs = np.arange(pcl.shape[1])
        np.random.shuffle(shuffle_idxs)
        pcl, labels = pcl[:, shuffle_idxs, :], labels[:, shuffle_idxs]
        return pcl, labels
    
    def random_rotation(self, pcl):
        x_angle, y_angle, z_angle = np.random.random() * 2 * np.pi * (np.random.random() < self.rot_prob[0]), \
                                    np.random.random() * 2 * np.pi * (np.random.random() < self.rot_prob[1]), \
                                    np.random.random() * 2 * np.pi * (np.random.random() < self.rot_prob[2])
        rotation = Rotation.from_euler('xyz', [x_angle, y_angle, z_angle], degrees=False)
        rotation_matrix = rotation.as_matrix()
        pcl = np.dot(pcl, rotation_matrix.T)
        return pcl
    
    def apply_augmentation(self, pcl, labels):
        pcl = self.random_rotation(pcl)
        pcl, labels = self.remove_points(pcl, labels)
        pcl = self.add_noise(pcl)
        pcl, labels = self.add_random_points(pcl, labels)
        return pcl, labels

if __name__ == '__main__':
    augmentator = Augmentator(remove_p=0.0, noise_std=0.00, add_points_p=0.0, rot_prob=[0.0, 0.0, 0.0])
    data_file = h5py.File('/home/fdimitri/workspace/Thesis/Thesis_Detection/datasets/JRDB/test_data.h5py', 'r')
    splitted_pcl, splitted_ann, centers = data_file['pcls'][0:10], \
                                        data_file['annotations'][0:10], \
                                        data_file['centers'][0:10]
    splitted_pcl, splitted_ann = augmentator.apply_augmentation(splitted_pcl, splitted_ann)
    splitted_ann = splitted_ann > 0
    for i in range(10):
        i = 5
        print(splitted_pcl.shape, splitted_ann.shape)
        ax = plt.subplot(1, 1, 1, projection='3d')
        ax.scatter(splitted_pcl[i, ~splitted_ann[i], 0], splitted_pcl[i, ~splitted_ann[i], 1], splitted_pcl[i, ~splitted_ann[i], 2], c='blue')
        ax.scatter(splitted_pcl[i, splitted_ann[i], 0], splitted_pcl[i, splitted_ann[i], 1], splitted_pcl[i, splitted_ann[i], 2], c='red')
        plt.show()
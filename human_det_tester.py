import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.pcl_utils import load_pcl, pcl_voxel, split_point_cloud_adaptive
from models.Pointnet import PointNetSeg
from utils.humanDBLoader import humanDBLoader
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.spatial import KDTree
from models.Pointnet2 import Pointet2

# reproducability
random_seed = 0 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)


def get_f1_score(predicted, ground_truth):
    f1_score = 0
    predicted = predicted.detach().cpu().numpy() > 0.5
    ground_truth = ground_truth.detach().cpu().numpy() > 0.5
    for (yout, yground) in zip(predicted, ground_truth):
        true_pos = (yout & yground).sum()
        false_neg = ((~yout) & yground).sum()
        false_pos = (yout & (~yground)).sum()
        prec = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        rec = true_pos / (true_pos + false_pos) if(true_pos + false_pos) > 0 else 0
        f1_score += 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return f1_score

if __name__ == '__main__':
    pcl_arr = ['./datasets/plane_detection_dataset/jrdb3.bin']
    model = Pointet2()
    model.load_state_dict(torch.load('./results/E35_v00.04.pt'))
    model.to('cuda')
    model.eval()
    for pcl_file in pcl_arr:
        pcl = load_pcl(pcl_file)
        pcl = pcl[np.linalg.norm(pcl, axis=1) < 15]
        pcl = pcl_voxel(pcl, 0.13)
        boxes, centers = split_point_cloud_adaptive(pcl, 2048, \
                                                    min_cluster_size=30, max_size_core=0.5, move_center=True)
        boxes_tor = torch.tensor(boxes).type(torch.cuda.FloatTensor).to('cuda')
        yout,_ = model(boxes_tor)
        yout = yout.detach().cpu().numpy()
        tree = KDTree(pcl)
        annots = np.zeros((pcl.shape[0], ))
        times = np.zeros((pcl.shape[0], ))
        for i, box in enumerate(boxes):
            box += centers[i]
            dists, idxs = tree.query(box, k=1)
            idxs = idxs.reshape(-1, )
            annots[idxs] += yout[i].reshape(-1, )
            times[idxs] += 1
        annots[times>0] /= times[times>0]
        annots = np.clip(annots, 0, 1)
        # annots = annots > 0.5
        ax = plt.subplot(1, 1, 1, projection='3d')
        ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], c=annots)
        plt.show()
        
    



import numpy as np
import matplotlib.pyplot as plt
from plane_detector import readPlanes, gr_planes_voxel, compute_local_pca
from o3d_funcs import load_pcl, o3d_to_numpy, plot_frame, pcl_voxel
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation


class robustNormalEstimator:
    def estimate_itter_monte_carlo(self, prob_in_det, out_rate):
        num_itter = np.log10(1 - prob_in_det) / np.log10(1 - (1 - out_rate)**3)
        return np.int32(num_itter)

    def get_max_con_sub(self, neibs_set, prob_in_det=0.99, out_rate=0.03):
        num_itter = self.estimate_itter_monte_carlo(prob_in_det, out_rate)
        min_set = []
        min_var = 1e10
        out_n = []
        for i in range(num_itter):
            h0 = neibs_set[np.random.choice(neibs_set.shape[0], 3, replace=False)]
            n = np.cross(h0[0] - h0[1], h0[0] - h0[2])
            n = n / np.linalg.norm(n)
            l0 = np.dot((neibs_set - np.mean(neibs_set, axis=0)), n.reshape(3, 1)).var()
            if l0 < min_var:
                min_set = h0
                min_var = l0
                out_n = n
        return min_set, out_n

    def remove_outlier(self, neibs_set, neib_sub, plane_nrm):
        ods = np.dot((neibs_set - np.mean(neib_sub, axis=0)), plane_nrm.reshape(3, 1))
        rzs = np.abs((ods - np.median(ods))) / (1.4826 * np.median(np.abs(ods - np.median(ods))) + 1e-10)
        filtered_p = neibs_set[rzs.reshape(-1, ) < 2.5]
        return filtered_p

    def robustNormalEstimation(self, points, k=10):
        tree = KDTree(points)
        normals = np.zeros((points.shape[0], 3))
        distances = np.zeros((points.shape[0], ))
        eigv_ratio = np.zeros((points.shape[0], )) + 1
        for i, point in enumerate(points):
            dists, neib_idxs = tree.query([point], k=k)
            distances[i] = np.median(dists)
            neib_idxs = neib_idxs.reshape(-1, )
            #neib_idxs = neib_idxs[neib_idxs != i]
            neibs_set = points[neib_idxs]
            neib_sub, plane_nrm = self.get_max_con_sub(neibs_set)
            filtered_neibs = self.remove_outlier(neibs_set, neib_sub, plane_nrm) - point
            if filtered_neibs.shape[0] < 5:
                continue
            pca = PCA(n_components=3)
            pca.fit(filtered_neibs)        
            normals[i, :] = pca.components_[np.argmin(pca.explained_variance_)]
            eigv_ratio[i] = np.min(pca.explained_variance_) / np.sum(pca.explained_variance_)
        return normals, distances, eigv_ratio

class customDetector:
    def __init__(self):
        self.k = 15
        self.sim_th = 0.95
        self.cand_score = 0.01
        self.rpca = robustNormalEstimator()
    
    def detectPlanes(self, pcl):
        normals, dists, eigv_ratio = self.rpca.robustNormalEstimation(pcl, self.k)
        #normals, eigv_ratio = compute_local_pca(pcl, 0.6, min_p=10)
        direc = np.sum(-pcl * normals)
        normals[direc < 0] *= -1
        groups, uncls = self.generateRegions(pcl, normals, eigv_ratio, dists)
        return groups
        
    def normal_test(self, group, cands):
        group_normal = np.median(group, axis=0).reshape(3, 1)
        group_normal = group_normal / (np.linalg.norm(group_normal)  + 1e-10)
        cands = cands / (np.linalg.norm(cands, axis=1)).reshape(-1, 1)
        similarities = np.abs(np.dot(cands, group_normal)).reshape(-1, )
        return similarities > self.sim_th

    # def points_plane_test(self, group_points, group_normals, cands_points):
    #     group_normal = np.median(group_normals, axis=0).reshape(3, 1)
    #     group_normal = group_normal / (np.linalg.norm(group_normal) + 1e-10)
    #     group_center = np.median(group_points, axis=0)
    #     projection_group = np.abs(np.dot(group_points - group_center, group_normal).reshape(-1, ))
    #     projection_cand = np.abs(np.dot(cands_points - group_center, group_normal).reshape(-1, ))
    #     rz_scores = np.abs((projection_cand - np.median(projection_group))) / (1.4826 * np.median(np.abs(projection_group - np.median(projection_group))))
    #     return rz_scores < 2.5

    def generateRegions(self, pcl, normals, eigv_ratio, distances):
        uncls = np.array([i for i in range(pcl.shape[0]) if (np.abs(normals[i]).sum() > 0) and (eigv_ratio[i] < self.cand_score)])
        uncls = np.argsort(distances)
        groups = []
        while uncls.shape[0] > self.k:
            new_group = np.array([uncls[0]])
            uncls = np.delete(uncls, 0)
            #tree = KDTree(pcl[uncls])
            dists_accum = np.array([], dtype=float)
            while uncls.shape[0] > self.k:
                tree = KDTree(pcl[uncls])
                dists, neibs = tree.query(pcl[new_group], k=self.k)
                dists, neibs = dists.reshape(-1, ), neibs.reshape(-1, )
                neibs_un = np.unique(neibs).astype('int')
                dists_un = np.array([dists[neibs.astype('int') == neibs_cand][0] for neibs_cand in neibs_un])
                dists_accum = np.append(dists_accum, dists_un)
                # neibs = np.unique(np.concatenate(tree.query(pcl[new_group], k=self.k)[1]).reshape(-1, ))
                if neibs_un.shape[0] == 0: break
                neibs = neibs_un[dists_un < np.quantile(dists_accum, 0.4)]
                normal_t = self.normal_test(normals[new_group].reshape(-1, 3), normals[uncls[neibs]].reshape(-1, 3))
                #points_plane_t = self.normal_test(normals[new_group].reshape(-1, 3), normals[uncls[neibs]].reshape(-1, 3))
                new_group = np.append(new_group, uncls[neibs[normal_t]])
                uncls = np.delete(uncls, neibs[normal_t])
                if (not (normal_t).sum()): break
            if(len(new_group) > self.k): groups.append(new_group)
        return groups, uncls

if __name__ == '__main__':
    pcl = o3d_to_numpy(load_pcl('../datasets/plane_detection_dataset/jrdb3.bin'))
    pcl = pcl[np.linalg.norm(pcl, axis=1)<20, :]
    #rot_mat = Rotation.from_euler('xyz', [45, 0, 0]).as_matrix()
    #pcl = np.dot(pcl, rot_mat.T)
    pcl = pcl_voxel(pcl, 0.1)
    det = customDetector()
    groups = det.detectPlanes(pcl)
    ax = plt.subplot(1, 1, 1, projection='3d')
    for group in groups:
        # ax = plt.subplot(1, 1, 1, projection='3d')
        ax.scatter(pcl[group, 0], pcl[group, 1], pcl[group, 2])
        ax.scatter(pcl[group[0], 0], pcl[group[0], 1], pcl[group[0], 2], s=100)
        # plt.show()
    plt.show()

    
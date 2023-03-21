import numpy as np
import matplotlib.pyplot as plt
from plane_detector import readPlanes, gr_planes_voxel, compute_local_pca, Plane, plot_plane_area
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
            for j in range(1, 4):
                dists, neib_idxs = tree.query([point], k= k * j)
                distances[i] = np.max(dists)
                neib_idxs = neib_idxs.reshape(-1, )
                #neib_idxs = neib_idxs[neib_idxs != i]
                neibs_set = points[neib_idxs]
                neib_sub, plane_nrm = self.get_max_con_sub(neibs_set)
                filtered_neibs = self.remove_outlier(neibs_set, neib_sub, plane_nrm) - point
                if filtered_neibs.shape[0] > k / 2:
                    break
            if filtered_neibs.shape[0] < 3:
                    continue
            pca = PCA(n_components=3)
            pca.fit(filtered_neibs)        
            normals[i, :] = pca.components_[np.argmin(pca.explained_variance_)]
            eigv_ratio[i] = np.min(pca.explained_variance_) / np.sum(pca.explained_variance_)
        direc = np.sum(-points * normals, axis=1)
        normals[direc < 0] *= -1
        return normals, distances, eigv_ratio

class customDetector:
    def __init__(self):
        self.k = 10
        self.sim_th = 0.95
        self.cand_score = 0.01
        self.rpca = robustNormalEstimator()
    
    def detectPlanes(self, pcl):
        normals, dists, eigv_ratio = self.rpca.robustNormalEstimation(pcl, self.k)
        groups = self.generateRegions(pcl, normals, eigv_ratio, dists)
        planes = self.estimatePlanes(pcl, normals, groups)
        return planes
        
    def normal_test(self, group, cands):
        group_normal = np.median(group, axis=0).reshape(3, 1)
        group_normal = group_normal / np.linalg.norm(group_normal)
        cands = cands / (np.linalg.norm(cands, axis=1)).reshape(-1, 1)
        similarities = np.abs(np.dot(cands, group_normal)).reshape(-1, )
        return similarities > self.sim_th

    def points_plane_test(self, group_points, group_normals, cands_points):
        group_normal = np.median(group_normals, axis=0).reshape(3, 1)
        group_normal = group_normal / (np.linalg.norm(group_normal) + 1e-10)
        group_center = np.median(group_points, axis=0)
        projection_group = np.abs(np.dot(group_points - group_center, group_normal).reshape(-1, ))
        projection_cand = np.abs(np.dot(cands_points - group_center, group_normal).reshape(-1, ))
        rz_scores = np.abs((projection_cand - np.median(projection_group))) / (1.4826 * np.median(np.abs(projection_group - np.median(projection_group))+ 1e-10))
        return rz_scores < 2.5

    def point_to_point_test(self, group_points, cands_points):
        tree = KDTree(group_points)
        group_dists, _ = tree.query(group_points, k=2)
        group_dists = group_dists[:, 1].reshape(-1, )
        cands_dists, _ = tree.query(cands_points, k=1)
        cands_dists = cands_dists.reshape(-1, )
        rz_scores = np.abs((cands_dists - np.median(group_dists))) / (1.4826 * np.median(np.abs(group_dists - np.median(group_dists))+ 1e-10))
        return rz_scores < 2.5


    def generateRegions(self, pcl, normals, eigv_ratio, distances):
        uncls = np.array([i for i in range(pcl.shape[0]) if (np.abs(normals[i]).sum() > 0) and (eigv_ratio[i] < self.cand_score)])
        uncls = np.argsort(distances)
        #distances = distances[uncls]
        groups = []
        while uncls.shape[0] > self.k:
            new_group = np.array([uncls[0]])
            uncls = np.delete(uncls, 0)
            while uncls.shape[0] > self.k:
                tree = KDTree(pcl[uncls])
                if len(new_group) > 1:
                    radius = np.median(distances[new_group]) + 3 * 1.4826 * np.median(distances[new_group] - np.abs(np.median(distances[new_group])))
                    neibs = np.unique(np.concatenate(tree.query_radius(pcl[new_group], r=radius)).reshape(-1, ))
                else:
                    neibs = tree.query(pcl[new_group], k=self.k)[1].reshape(-1, )
                if neibs.shape[0] == 0: break
                test_th = self.normal_test(normals[new_group].reshape(-1, 3), normals[uncls[neibs]].reshape(-1, 3))
                if len(new_group) > 1:
                    test_th &= self.points_plane_test(pcl[new_group], normals[new_group].reshape(-1, 3), pcl[uncls[neibs]].reshape(-1, 3))
                new_group = np.append(new_group, uncls[neibs[test_th]])
                uncls = np.delete(uncls, neibs[test_th])
                if (not (test_th).sum()): break
            if(len(new_group) > self.k): groups.append(new_group)
        return groups

    def estimatePlanes(self, pcl, normals, groups):
        planes = []
        clsf_idxs = np.concatenate(groups)
        uclsf_idxs = np.array([i for i in range(pcl.shape[0]) if i not in clsf_idxs], dtype='int')
        clsf_cls = np.concatenate([[i]*len(group) for i, group in enumerate(groups)])
        group_lens = [len(group) for group in groups]
        sorted_group_idxs = np.argsort(group_lens)
        group_status = [0 for i in groups]
        uncls_idxs = np.arange(pcl.shape[0])
        for g_idxs in sorted_group_idxs:
            curr_group = groups[g_idxs]
            curr_plane = Plane(inliers=[], normal=np.median(normals[curr_group], axis=0))
            uncls_idxs = np.delete(uncls_idxs, [i for i, idx in enumerate(uncls_idxs) if idx in curr_group])
            while True:
                cand_idxs = uncls_idxs[self.points_plane_test(pcl[curr_group], normals[curr_group], pcl[uncls_idxs])]
                if(cand_idxs.shape[0] == 0): break
                passed_idxs = cand_idxs[self.point_to_point_test(pcl[curr_group], pcl[cand_idxs])]
                if(passed_idxs.shape[0] == 0): break
                p_uclsf_idxs = np.intersect1d(uclsf_idxs, passed_idxs).reshape(-1, )
                if(p_uclsf_idxs.shape[0] == 0): break
                curr_group = np.append(curr_group, p_uclsf_idxs)
                uncls_idxs = np.delete(uncls_idxs, [i for i, idx in enumerate(uncls_idxs) if idx in p_uclsf_idxs])
            curr_plane.inliers = curr_group
            planes.append(curr_plane)
        return planes

if __name__ == '__main__':
    pcl = o3d_to_numpy(load_pcl('../datasets/plane_detection_dataset/box.bin'))
    pcl = pcl[np.linalg.norm(pcl, axis=1)<20, :]
    #rot_mat = Rotation.from_euler('xyz', [45, 0, 0]).as_matrix()
    #pcl = np.dot(pcl, rot_mat.T)
    pcl = pcl_voxel(pcl, 0.2)
    det = customDetector()
    planes = det.detectPlanes(pcl)
    plot_plane_area(pcl, planes)

import numpy as np
import matplotlib.pyplot as plt
from plane_detector import readPlanes, gr_planes_voxel, compute_local_pca, Plane, plot_plane_area
from o3d_funcs import o3d_to_numpy, plot_frame, pcl_voxel
from pcl_utils import load_pcl
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation
from time import time
from multiprocessing import Pool, cpu_count
from sklearn.linear_model import RANSACRegressor


class robustNormalEstimator:
    """
    This is a robust normal estimator class that removes outliers and then calculates
    PCA  
    """
    def estimate_itter_monte_carlo(self, prob_in_det, out_rate):
        """
        Calculates the number of itterations needed to find the best minimal subset
        from which the best fitted plane can be calulcated. It uses mode carclo probabilistic 
        approach.

        @param prob_in_det:  The required probability of fidning the best minimal subset
        @param out_rate: outliers probability in dataset
        @return: the number of iterations
        """
        num_itter = np.log10(1 - prob_in_det) / np.log10(1 - (1 - out_rate)**3)
        return np.int32(num_itter)

    def get_max_con_sub(self, neibs_set, prob_in_det=0.99, out_rate=0.25):
        """
        Gets the maximum consistent set of a neighborhood of points. The maximum consistent
        set of half of points that can be best expressed with a plane.
        
        @param neibs_set: the neighborhood set
        @param prob_in_det:  The required probability of fidning the best minimal subset
        @param out_rate: outliers probability in dataset
        @return: the maximum consistent set and the normal of the best fitted plane
        """
        num_itter = self.estimate_itter_monte_carlo(prob_in_det, out_rate)
        min_set = np.zeros((neibs_set.shape[0], 5, 3))
        min_var = np.zeros(neibs_set.shape[0]) + 100
        out_normals = np.zeros((neibs_set.shape[0], 3))
        for i in range(num_itter):
            h0 = neibs_set[:,np.random.choice(neibs_set.shape[1], 3, replace=False), :]      #h0: pick random 3 points
            n = np.cross(h0[:, 0] - h0[:, 1], h0[:, 0] - h0[:, 2])              # normal vector of h0
            n = n / np.linalg.norm(n, axis=1, keepdims=True)                   # normalie normal vector of h0
            ods = np.einsum('ijk,ik->ij', (neibs_set - np.mean(h0, axis=1, keepdims=True)), n)   # calculate the projections
            max_set_cand_idxs = np.argsort(ods, axis=1)[:, :ods.shape[1] // 2]     # sort them and keep half
            max_set_cand =  neibs_set[np.arange(len(max_set_cand_idxs))[:, None], max_set_cand_idxs]
            max_set_cand = max_set_cand - np.mean(max_set_cand, axis=1, keepdims=True)
            num_workers = min(cpu_count(), len(max_set_cand))
            with Pool(num_workers) as p:
                pca_results = np.concatenate(p.map(self.pca_worker, max_set_cand), axis=0)
            eigvalues = pca_results[:, 0]
            eigvec = pca_results[:, 1:]
            min_set[eigvalues < min_var] = max_set_cand[eigvalues < min_var]
            out_normals[eigvalues < min_var] = eigvec[eigvalues < min_var]
            min_var[eigvalues < min_var] = eigvalues[eigvalues < min_var]
        return min_set, out_normals
    
    def ransac_plane(self, points):
        ransac = RANSACRegressor(min_samples=3, max_trials=6)
        ransac.fit(points[:, :2], points[:, 2])
        normal_vec = np.array([ransac.estimator_.coef_[0], ransac.estimator_.coef_[1], -1])
        distances = np.abs(points[:, 0] * normal_vec[0] + points[:, 1] * normal_vec[1] + points[:, 2] * normal_vec[2] + \
            ransac.estimator_.intercept_) / np.linalg.norm(normal_vec)
        resutls = np.zeros((1, 3+points.shape[0]//2))
        resutls[0, :3] = normal_vec
        resutls[0, 3:] = distances.argsort()[:points.shape[0]//2]
        return resutls

    def get_max_con_sub_v2(self, neibs_set):
        """
        Gets the maximum consistent set of a neighborhood of points. The maximum consistent
        set of half of points that can be best expressed with a plane.
        
        @param neibs_set: the neighborhood set
        @param prob_in_det:  The required probability of fidning the best minimal subset
        @param out_rate: outliers probability in dataset
        @return: the maximum consistent set and the normal of the best fitted plane
        """
        num_workers = min(cpu_count(), len(neibs_set))
        with Pool(num_workers) as p:
            results = np.concatenate(p.map(self.ransac_plane, neibs_set), axis=0)
        out_normals = results[:, :3]
        min_set = neibs_set[np.arange(len(results))[:, None], results[:, 3:].astype('int')]
        return min_set, out_normals

    def get_outlier(self, neibs_set, neib_sub, plane_nrm):
        """
        Uses Robust-z score based on robust statistics to identify outliers.

        @param neibs_set: the neighbohood of points
        @param neib_sub: the maximum consisten subset
        @param plane_nrm: outliers probability in dataset
        @return: the maximum consistent set and the normal of the best fitted plane
        """
        ods = np.einsum('ijk,ik->ij', (neibs_set - np.mean(neib_sub, axis=1, keepdims=True)), plane_nrm)
        rzs = np.abs((ods - np.median(ods, axis=1, keepdims=True))) / (1.4826 * np.median(np.abs(ods - np.median(ods, axis=1, keepdims=True)), axis=1, keepdims=True) + 1e-10)
        return rzs > 2.5
    
    def pca_worker(self, arr):
        # pca = PCA(n_components=3, svd_solver='full')
        # pca.fit(arr)
        # results = np.zeros((1, 4))
        # results[0, 0], results[0, 1:] = pca.explained_variance_.min()/pca.explained_variance_.sum(), \
        #      pca.components_[np.argmin(pca.explained_variance_), :]
        cov = np.cov(arr, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        min_explained_variance_ratio = np.min(explained_variance_ratio)
        min_eigenvector = eigenvectors[:, np.argmin(explained_variance_ratio)]
        results = np.zeros((1, 4))
        results[0, 0] = min_explained_variance_ratio / np.sum(explained_variance_ratio)
        results[0, 1:] = min_eigenvector
        return results
    
    def robustNormalEstimation(self, points, k=10):
        tree = KDTree(points)
        normals = np.zeros((points.shape[0], 3))
        distances = np.zeros((points.shape[0], ))
        eigv_ratio = np.zeros((points.shape[0], )) + 1
        dists, neib_idxs = tree.query(points, k=k)
        distances = np.max(dists, axis=1)
        neibs_set = points[neib_idxs]
        neib_sub, plane_nrm = self.get_max_con_sub(neibs_set)
        outliers = self.get_outlier(neibs_set, neib_sub, plane_nrm)
        # merge into a list each neighborhood
        list_arr = []
        for i, point in enumerate(points):
            list_arr.append(neibs_set[i][~outliers[i]] - point)
        # parallel calculation of PCA for evey neighborhood
        num_workers = min(cpu_count(), len(list_arr))
        with Pool(num_workers) as p:
            pca_results = np.concatenate(p.map(self.pca_worker, list_arr), axis=0)
        eigv_ratio = pca_results[:, 0]
        normals = pca_results[:, 1:]
        direc = np.sum(-points * normals, axis=1)
        normals[direc < 0] *= -1
        return normals, distances, eigv_ratio

class customDetector:
    def __init__(self):
        self.k = 10
        self.sim_th = 0.90
        self.cand_score = 0.01
        self.rpca = robustNormalEstimator()
    
    def detectPlanes(self, pcl):
        t0 = time()
        normals, dists, eigv_ratio = self.rpca.robustNormalEstimation(pcl, self.k)
        t1 = time()
        print(t1 - t0)
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
                    radius = np.median(distances[new_group]) + 2.5 * 1.4826 * np.median(distances[new_group] - np.abs(np.median(distances[new_group])))
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
    pcl = load_pcl('../datasets/plane_detection_dataset/bedroom64.bin')
    pcl = pcl[np.linalg.norm(pcl, axis=1)<20, :]
    pcl = pcl_voxel(pcl, 0.3)
    det = customDetector()
    t0 = time()
    planes = det.detectPlanes(pcl)
    t1 = time()
    print(t1 - t0, pcl.shape)
    plot_plane_area(pcl, planes)

    # A, B, C, D = 1, 3, 4, 10
    # x, y = np.random.rand(100)*5, np.random.rand(100)*5
    # z = - A/C* x - B/C * y - D
    # input_arr = np.stack([x, y, z], axis=1)
    # rest = robustNormalEstimator()
    # normals, dists, eigv_ratio = rest.robustNormalEstimation(input_arr, 10)
    # ax = plt.subplot(1, 1, 1, projection = '3d')
    # ax.scatter(x, y, z)
    # ax.quiver(input_arr[:, 0], input_arr[:, 1], input_arr[:, 2], normals[:, 0], normals[:, 1], normals[:, 2])
    # plt.show()

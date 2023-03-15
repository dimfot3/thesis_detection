import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA


def estimate_itter_monte_carlo(prob_in_det, out_rate):
    num_itter = np.log10(1 - prob_in_det) / np.log10(1 - (1 - out_rate)**3)
    return np.int32(num_itter)

def get_max_con_sub(neibs_set, prob_in_det=0.99, out_rate=0.3):
    num_itter = estimate_itter_monte_carlo(prob_in_det, out_rate)
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

def remove_outlier(neibs_set, neib_sub, plane_nrm):
    ods = np.dot((neibs_set - np.mean(neib_sub, axis=0)), plane_nrm.reshape(3, 1))
    rzs = np.abs((ods - np.median(ods))) / (1.4826 * np.median(np.abs(ods - np.median(ods))))
    filtered_p = neibs_set[rzs.reshape(-1, ) < 2.5]
    return filtered_p
from sklearn.covariance import MinCovDet

def remove_outlier_v2(neibs_set, neib_sub, plane_nrm):
    filtered_p = np.array([]).reshape(-1, 3)
    centered_neibs = neibs_set - neib_sub.mean(axis=0)
    cov_mat = 1/(neib_sub.shape[0]-1) * (neib_sub - neib_sub.mean(axis=0)).T * (neib_sub - neib_sub.mean(axis=0))
    mcd = MinCovDet().fit(neib_sub)
    rmd = mcd.mahalanobis(neibs_set)
    filtered_p = neibs_set[rmd < 3.075]
    return filtered_p

def robustNormalEstimation(points, k=10):
    tree = KDTree(points)
    normals = np.zeros((points.shape[0], 3))
    for i, point in enumerate(points):
        _, neib_idxs = tree.query([point], k=k)
        neib_idxs = neib_idxs.reshape(-1, )
        #neib_idxs = neib_idxs[neib_idxs != i]
        neibs_set = points[neib_idxs]
        neib_sub, plane_nrm = get_max_con_sub(neibs_set)
        filtered_neibs = remove_outlier(neibs_set, neib_sub, plane_nrm) - point
        if filtered_neibs.shape[0] < 3:
            continue
        pca = PCA(n_components=3)
        pca.fit(filtered_neibs)        
        normals[i, :] = pca.components_[np.argmin(pca.explained_variance_)]
    return normals

if __name__ == '__main__':
    n = 100
    noise_std = 2
    np.random.seed(42)
    points = - 5 + np.random.random((n, 3)) * 10
    a,b,c,d = 2, 0, 2, 1
    gr_normal = np.array([a, b, c]) / np.sqrt(a**2 + b**2 + c**2)
    points[:, 2] = - (a * points[:, 0] +  b * points[:, 1] + d) / c
    noise = - noise_std / 2 + np.random.random((n, 3)) * noise_std
    points += noise
    outliers = - 5 + np.random.random((20, 3)) * 10
    points = np.append(points, outliers, axis=0)

    normals = robustNormalEstimation(points, k=20)
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    print(np.mean(np.abs(np.dot(normals[:n], gr_normal.reshape(-1, 1)))))
    for point, normal in zip(points, normals):
        ax.quiver(*point, *normal)
    plt.show()
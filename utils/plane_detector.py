import numpy as np
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor, LinearRegression
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from functools import reduce
import struct
import ezdxf
from scipy.spatial import ConvexHull

class Plane:
    """
    Plane is a structure that holds the inlier idxs and normal of a plane
    """
    def __init__(self, inliers=[], normal=None):
        self.normal = normal
        self.inliers = inliers

def compute_local_pca(pointcloud, radius, min_p=10):
    """
    Computes the principal component analysis (PCA) for each point in a given point cloud within a certain radius.

    :param pointcloud: A numpy array of shape (num_points, 3) representing the input point cloud where each row
                       contains the (x, y, z)
    :param min_p: the minimum number of points for a valid pca in local area.
    :param radius: The radius of the local region to compute the PCA within.
    :return: A tuple of numpy arrays representing the computed eigenvectors and eigenvalues of the PCA, where the
             eigenvectors array has shape (num_points, 3) and each row contains the unit normal vector of the plane
             fitted to the local region around the corresponding point in the input cloud, and the eigenvalues array
             has shape (num_points,) and contains the minimum eigenvalue of the local covariance matrix for each point.
             In points where pca where not successful, eigen vectors and eigen values are all zero.
    """
    num_points = pointcloud.shape[0]
    eigenvectors = np.zeros((num_points, 3))
    eigenvalues = np.zeros((num_points, ))
    if(num_points == 0): return eigenvectors, eigenvalues       # handle the case of empty pcl

    min_p = max(min_p, 3)
    # Build a KD-tree for fast nearest neighbor search
    tree = KDTree(pointcloud[:, :3])
    # Compute the PCA for each point
    for i in range(num_points):
        # Find the nearest neighbors within a certain radius
        neighbors_idx = tree.query_radius(pointcloud[i, :3].reshape(1, -1), r=radius)[0]
        if(np.unique(pointcloud[neighbors_idx], axis=0).shape[0] < min_p):
            continue
        # Extract the local point cloud and center it around the current point
        local_points = pointcloud[neighbors_idx, :3] - pointcloud[i, :3]
        # Compute the PCA of the local point cloud
        pca = PCA(n_components=3)
        pca.fit(local_points)
        normal_verctor = pca.components_[np.argmin(pca.explained_variance_)]
        # Store the eigenvectors and eigenvalues of the PCA
        dot_product = np.sign(np.dot(-pointcloud[i, :3], normal_verctor))
        eigenvectors[i, :] = normal_verctor if dot_product >= 0 else -normal_verctor
        eigenvalues[i] = np.min(pca.explained_variance_)
    return eigenvectors, eigenvalues

def classify_points(point_vectors):
    """
    classify_points function classifies each point in a pointcloud into one of four categories - floor, ceiling, vertical and other.

    :param point_vectors: A numpy array of size (n, 3) containing the eigenvectors of each point in the pointcloud.
    :param eigenvalues: A numpy array of size (n,) containing the eigenvalues of each point in the pointcloud.
    :param theta_vert_err: the angle error thrshold to be considered wall
    :param theta_hor_err: the angle error thrshold to be considered floor or ceil
    :param eigenvalmin: Minimum eigenvalue for considering a point to be a feature point.
    :return: A numpy array of size (n,) containing the class labels of each point. Class labels: 0 - horizontal, 1 - vertical plane (x), 2-vertical plane (y)
    """
    if point_vectors.shape[0] == 0: return np.array([], dtype='int')        # hangle empty input
    valid_idxs = (np.abs(point_vectors).sum(axis=1) != 0)
    point_vectors[valid_idxs] = point_vectors[valid_idxs] / np.linalg.norm(point_vectors[valid_idxs], axis=1).reshape(-1, 1)       # normalize eigen vectors
    normalz = np.array([[0, 0, -1], [0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    dot_prods = np.zeros((point_vectors.shape[0], 6))
    dot_prods[valid_idxs] = np.dot(point_vectors[valid_idxs], normalz.T)
    thetas = np.arccos(dot_prods) * 180 / np.pi
    class_arr = np.zeros((point_vectors.shape[0], ), dtype='int') - 1
    class_arr[valid_idxs & ((thetas[:, 2] < 45) | (thetas[:, 3] < 45))  ] = 2         # wallX
    class_arr[valid_idxs & ((thetas[:, 4] < 45) | (thetas[:, 5] < 45))  ] = 3          # wallY
    class_arr[valid_idxs & (thetas[:, 1] < 15) ] = 1        # floor
    class_arr[valid_idxs & (thetas[:, 0] < 15) ] = 0        # ceiling
    return class_arr

def generate_regions(points, classes, distx, k):
    """
    generate_regions is a function that generates regions of interest based on the given point cloud and class labels. 
    It uses a KD-tree to efficiently find neighboring points and groups them together into regions based on their 
    class label and proximity to each other.

    :param points: numpy array of shape (N, 3) representing the coordinates of the points in the point cloud
    :param classes: numpy array of shape (N, ) representing the class labels for each point in the point cloud
    :param distx: float representing the maximum distance between two points for them to be considered part of 
    the same region
    :param k: integer representing the minimum number of points required for a region to be considered valid
    :return: two lists, regions and class_regions. regions is a list of lists, where each sublist contains the 
    indices of the points belonging to a single region. class_regions is a list containing the class label of 
    each region in the same order as they appear in the regions list.
    """
    points_queue = np.arange(points.shape[0])
    region_arr = []
    class_arr = []
    while points_queue.shape[0] > 0:
        root = points_queue[0]
        cur_class = classes[root]
        points_queue = np.delete(points_queue, 0)
        neibs = points_queue[(cdist([points[root]], points[points_queue], 'euclidean')[0] < distx) & (classes[points_queue] == cur_class)]
        points_queue = np.delete(points_queue, np.isin(points_queue, neibs))
        if (len(neibs) == 0) or (cur_class == -1):
            continue
        else:
            region = np.append(neibs, root).reshape(-1, )
            tree_reg = KDTree(points[region])
            points_queue_can = points_queue[classes[points_queue] == cur_class]
            while points_queue_can.shape[0] > 0:
                can_idxs = tree_reg.query_radius(points[points_queue_can], r=distx)
                pos_can = np.array([True if arr.shape[0] > 0 else False for arr in can_idxs])
                if (pos_can==False).all():
                    break
                region = np.append(region, points_queue_can[pos_can]).reshape(-1, )
                tree_reg = KDTree(points[region])
                points_queue_can = points_queue_can[~pos_can]
            points_queue = points_queue[~np.isin(points_queue, region)]
            if(len(region) >= k):
                region_arr.append(region)
                class_arr.append(cur_class)
    return region_arr, np.array(class_arr)

def estimate_plane_ransac(data, n_iterations=100, threshold=0.1):
    """
    estimate_plane_ransac estimates a plane from 3D data using RANSACRegressor and linear regression.
    :param data: A 2D numpy array of shape (n_samples, 3) containing the 3D data points.
    :param n_iterations: The maximum number of RANSAC iterations to perform.
    :param threshold: The maximum distance threshold between a data point and the estimated plane to consider it an inlier.
    :return: A tuple of the form (inliers, coefficients) where inliers is a boolean array of shape (n_samples,)
    indicating which data points are inliers and coefficients is a 1D numpy array of shape (3,) containing the 
    coefficients of the estimated plane in the form (a, b, d) where a*x + b*y + d = z is the equation of the plane.
    """
    # Initialize RANSAC estimator
    estimator = RANSACRegressor(LinearRegression(), min_samples=3, residual_threshold=threshold, max_trials=n_iterations, random_state=0)
    min_cov_idx = np.diag(np.cov(data.T)).argmin()
    other_idxs = np.argwhere(np.arange(data.shape[1]) != min_cov_idx).reshape(-1, )
    X = data[:, other_idxs]
    y = data[:, min_cov_idx]
    estimator.fit(X, y)
    # Get inliers and coefficients
    inliers = estimator.inlier_mask_
    # refine the coefficients
    reg = LinearRegression().fit(X[inliers], y[inliers])
    plane_normal = np.zeros((3, ))
    plane_normal[min_cov_idx] = -1
    plane_normal[other_idxs] = reg.coef_
    plane_normal /= np.linalg.norm(plane_normal)
    return inliers, plane_normal
    
def pca_plane_det(pcl, pca_radius=0.5, distmin=0.5, minp=15, ransac_iter=100, ransac_thres=0.1):
    eigv, eigenvalues = compute_local_pca(pcl, pca_radius)
    classes = classify_points(eigv)
    regions, class_regions = generate_regions(pcl, classes, distmin, minp)
    filter_idxs = np.argwhere(class_regions >= 0).reshape(-1, )
    regions = [regions[i] for i in filter_idxs]
    plane_inliers = []
    plane_normals = []
    for region in regions:
        inliers, plane_normal = estimate_plane_ransac(pcl[region], ransac_iter, ransac_thres)
        plane_inliers.append(region[inliers])
        plane_normals.append(plane_normal)
    return plane_inliers, plane_normals

def points_near_line_segment(points, p1, p2):
    # Calculate the line segment vector and its length squared
    v = p2 - p1
    len2 = np.sum(v**2)
    # If the length of the line segment is very small, return an empty array
    if len2 < 1e-8:
        return np.array([])
    # Calculate the parameter values along the line segment for each point
    params = np.dot(points - p1, v) / len2
    # Find the points whose parameter values are between 0 and 1
    mask = (params >= 0) & (params <= 1)
    # Calculate the minimum distance between each near point and the line segment
    dists = np.abs(np.cross(v, points - p1)) / np.sqrt(len2)
    dists[~mask] = np.inf
    return dists

def get_wall_points(pcl, lines, dthresh):
    accumulator = np.zeros((pcl.shape[0], len(lines)))
    for i, line in enumerate(lines):
        accumulator[:, i] = points_near_line_segment(pcl[:, :2], line[:2], line[2:4])
    min_dists = np.min(accumulator, axis=1)
    points_wall = min_dists <= dthresh
    return points_wall

def read_gr_lines(fpath):
    # Open the DXF file
    doc = ezdxf.readfile(fpath)
    # Get the modelspace entity of the DXF document
    modelspace = doc.modelspace()
    # Find the first polyline entity in the modelspace
    polyline = modelspace.query("POLYLINE").first
    # Get the vertex coordinates of the polyline
    vertices = np.array([[vertex.dxf.location.x, vertex.dxf.location.y] for vertex in polyline.vertices]).reshape(-1, 2)
    lines = np.array([[vertices[2*i, 0], vertices[2*i, 1], vertices[2*i+1, 0], vertices[2*i+1, 1]] for i in range(vertices.shape[0]//2)])
    return lines

def save_planes(planes, fileout):
    """
    save_planes saves a list with Plane elements in binary format
    is '{num_of_planes} (8bytes) {normal_x} (4byters) {normal_y} (4byters) {normal_z} (4byters) {num_of_inliers} (8bytes)
    {inlier_idx_1} (8bytes) ... {inlier_idx_n} (8bytes) {normal_x} (4byters) ...'
    :param planes: list of elements of type Plane
    :param fileout: the path for the output file
    """
    num_of_planes = len(planes)
    with open(fileout, 'wb') as f:
       data = struct.pack('Q', num_of_planes)
       f.write(data)
       for plane in planes:
            num_inliers = len(plane.inliers)
            format_str = '<3f {}Q'.format(num_inliers + 1)
            data = struct.pack(format_str, *plane.normal, num_inliers, *plane.inliers)
            f.write(data)

def readPlanes(file: str):
    """
    readPlanes reads a from a binary file a plane. The format of the plane 
    is '{num_of_planes} (8bytes) {normal_x} (4byters) {normal_y} (4byters) {normal_z} (4byters) {num_of_inliers} (8bytes)
    {inlier_idx_1} (8bytes) ... {inlier_idx_n} (8bytes) {normal_x} (4byters) ...'
    :param file: the path for the file
    :return: list of elements of type Plane
    """
    planes = []
    with open(file, 'rb') as f:
        # Read the number of circles and planes
        numPlanes, = struct.unpack('Q', f.read(8))
        # Read each plane
        for i in range(numPlanes):
            plane = Plane()
            plane.normal = struct.unpack('<3f', f.read(12))
            numInliers = struct.unpack('<Q', f.read(8))[0]
            format_str = '{}Q'.format(numInliers)
            plane.inliers = list(struct.unpack(format_str, f.read(numInliers * 8)))
            planes.append(plane)
    return planes

def visualize_planes(pcl, planes):
    ax = plt.subplot(1, 1, 1, projection='3d')
    #ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], c='black')
    for plane in planes:
        ax.scatter(pcl[plane.inliers, 0], pcl[plane.inliers, 1], pcl[plane.inliers, 2])
        ax.quiver(pcl[plane.inliers, 0].mean(), pcl[plane.inliers, 1].mean(), pcl[plane.inliers, 2].mean(), plane.normal[0], plane.normal[1], plane.normal[2], length=1)
    plt.show()

def project_points_to_plane(pcl, plane):
    points = pcl[plane.inliers]
    plane_center = pcl[plane.inliers].mean(axis=0)
    plane_normal = np.array(plane.normal)
    projection = np.dot(points - plane_center, plane_normal)[:, np.newaxis] * plane_normal
    projected_points = points - projection
    return projected_points

def compute_convex_hull_on_plane(pcl, planes):
    areas = []
    for plane in planes:
        projected_points = project_points_to_plane(pcl, plane)
        hull = ConvexHull(projected_points[:, :2])
        areas.append(projected_points[hull.vertices])
    return areas

def plot_plane_area(pcl, planes, areas):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2])
    inlier_max, inlier_min = 0, 0
    for i, area in enumerate(areas):
        ax.plot_trisurf(area[:, 0], area[:, 1], area[:, 2])
        ax.scatter(pcl[planes[i].inliers, 0], pcl[planes[i].inliers, 1], pcl[planes[i].inliers, 2])
        ax.scatter(pcl[planes[i].inliers, 0], pcl[planes[i].inliers, 1], pcl[planes[i].inliers, 2])
        # ax.quiver(pcl[planes[i].inliers, 0].mean(), pcl[planes[i].inliers, 1].mean(), pcl[planes[i].inliers, 2].mean(),\
        #            planes[i].normal[0], planes[i].normal[1], planes[i].normal[2], length=1.5, linewidths=5)
        inlier_max = np.max([inlier_max, np.abs(pcl[planes[i].inliers]).max()])

    plt.show()
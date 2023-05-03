import numpy as np
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor, LinearRegression
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from functools import reduce
from utils.o3d_funcs import pcl_voxel
import struct
import ezdxf
from scipy.spatial import ConvexHull
from utils.RobustNormalEstimation import robustNormalEstimation


class Plane:
    """
    Plane is a structure that holds the inlier idxs and normal of a plane
    """
    def __init__(self, inliers=[], normal=None):
        self.normal = normal
        self.inliers = inliers

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
        hull = ConvexHull(projected_points[:, :2], qhull_options='QG4')
        areas.append(projected_points[hull.vertices])
    return areas

def plot_plane_area(pcl, planes):
    areas = compute_convex_hull_on_plane(pcl, planes)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2])
    inlier_max, inlier_min = 0, 0
    for i, area in enumerate(areas):
        ax.plot_trisurf(area[:, 0], area[:, 1], area[:, 2])
        ax.scatter(pcl[planes[i].inliers, 0], pcl[planes[i].inliers, 1], pcl[planes[i].inliers, 2])
        ax.scatter(pcl[planes[i].inliers, 0], pcl[planes[i].inliers, 1], pcl[planes[i].inliers, 2])
        # ax.quiver(pcl[planes[i].inliers, 0].mean(), pcl[planes[i].inliers, 1].mean(), pcl[planes[i].inliers, 2].mean(),\
        #            (-1) * planes[i].normal[0], (-1) * planes[i].normal[1], (-1) * planes[i].normal[2], length=1.5, linewidths=5)
        inlier_max = np.max([inlier_max, np.abs(pcl[planes[i].inliers]).max()])
    plt.show()

def gr_planes_voxel(pcl, gr_planes, voxel_size=0.1):
    new_planes = []
    for gr_plane in gr_planes:
        pcl_down = pcl_voxel(pcl, voxel_size)
        tree = KDTree(pcl_down)
        _, ind = tree.query(pcl[gr_plane.inliers], k=1)
        new_idxs = np.unique(ind.reshape(-1, ))
        new_planes.append(Plane(list(new_idxs), gr_plane.normal))
    return new_planes
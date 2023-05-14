import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import ros2_numpy
from sklearn.cluster import DBSCAN


def merge_boxes_output(pcl, centers, yout, boxes):
    """
    @brief: merge boxes and keep maximum annotation for each common points
    """
    tree = KDTree(pcl)
    annots = np.zeros((pcl.shape[0], ))
    # times = np.zeros((pcl.shape[0], ))
    for i, box in enumerate(boxes):
        box += centers[i]
        _, idxs = tree.query(box, k=1)
        idxs = idxs.reshape(-1, )
        annots[idxs] = np.maximum(annots[idxs], yout[i].reshape(-1, ))
        # times[np.intersect1d(idxs, np.argwhere(yout[i] > 0.001).reshape(-1, ))] += 1
    # annots[times > 0] /= times[times > 0]
    pcl = np.hstack((pcl, annots.reshape(-1, 1))).astype('float32')
    return pcl

def tfmsg_to_matrix(tf_msg):
    """
    Creates a transformation matrix from the transformation message
    """
    # Extract the transform information from the message
    transform = tf_msg.transforms[0].transform
    translation = transform.translation
    rotation = transform.rotation
    translation_array = np.array([translation.x, translation.y, translation.z])
    rotation_array = np.array([rotation.x, rotation.y, rotation.z, rotation.w])
    rotation_matrix = Rotation.from_quat(rotation_array).as_matrix()[:3, :3]
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3], translation_matrix[:3, :3], translation_matrix[3, 3] = translation_array, \
                                                                        rotation_matrix, 1
    return translation_matrix

def pcl2_to_numpy(msg, tf=None):
    pc = ros2_numpy.numpify(msg)
    points= pc['xyz']
    points = points[np.all(np.isfinite(points), axis=1)]
    if(type(tf) != type(None)):
        points = np.dot(points, tf[:3, :3].T) + tf[:3, 3].T
    return points
    
def semantic_to_instance(pcl, det_thresh, u_det_thresh, min_human_p, max_dist_hum):
    clustering = DBSCAN(eps=max_dist_hum, min_samples=min_human_p).fit(pcl[:, :3])
    cluster_labels = clustering.labels_
    human_ids = np.unique(cluster_labels[cluster_labels>=0])
    human_poses = np.zeros(shape=(0, 3))
    for i, human_id in enumerate(human_ids):
        cluster_vec = cluster_labels == human_id
        if(pcl[cluster_vec, 3].mean() > np.clip(((u_det_thresh - det_thresh) / 80) * (pcl[cluster_vec, 3].shape[0] - min_human_p) + det_thresh, det_thresh, u_det_thresh)):
            human_poses = np.append(human_poses, pcl[cluster_vec, :3].mean(axis=0).reshape(1, 3), axis=0)
        else: cluster_labels[cluster_vec] = -1
    return human_poses, cluster_labels

def keep_k_det(pcl, human_poses, cluster_labels, K):
    """
    Kepps the K clusters with highest average prbability score
    """
    if(K == None): return human_poses, cluster_labels
    K = np.minimum(K, human_poses.shape[0])
    human_ids = np.unique(cluster_labels[cluster_labels>=0]).reshape(-1, )
    scores = np.zeros(human_ids.shape[0])
    for i, human_id in enumerate(human_ids):
        scores[i] = pcl[cluster_labels == human_id, 3].mean()
    sorted_idxs = np.argsort(scores)[::-1]
    human_poses = human_poses[sorted_idxs][:K]
    cluster_to_keep = human_ids[sorted_idxs][:K]
    rejected_points = ~np.in1d(cluster_labels, cluster_to_keep)
    cluster_labels[rejected_points] = -1
    return human_poses, cluster_labels
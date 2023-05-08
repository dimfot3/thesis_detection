import numpy as np
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import PointCloud2
from scipy.spatial.transform import Rotation
import threading
import matplotlib.pyplot as plt
import queue
from message_filters import ApproximateTimeSynchronizer, Subscriber
from time import time
import ros2_numpy
from utils.o3d_funcs import pcl_voxel
from utils.hierarchical_splitting import split_pcl_to_clusters, focused_split_to_boxes
from sensor_msgs.msg import PointCloud2, PointField
from models.Pointnet import PointNetSeg
from models.Pointnet2 import Pointet2
from geometry_msgs.msg import PointStamped
import torch
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from builtin_interfaces.msg import Time
from collections import Counter


class HumanDetector(Node):
    def __init__(self, lidar_list, det_model, max_hum=None):
        super().__init__('human_detector')
        self.lidar_list = lidar_list
        self.lidar_frames, self.lidar_pcl, self.lidar_times = {}, {}, {}
        self.det_model = det_model
        self.tf_sub = self.create_subscription(TFMessage, '/tf', self.read_lidar_frames, 20)
        self.lidar_sub = {}
        for lidar in lidar_list:
            self.lidar_sub[lidar] = Subscriber(self, PointCloud2, lidar)
        self.tf_frames_event = threading.Event()
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.curr_lidar = queue.Queue(maxsize=1)
        self.pub_cls = self.create_publisher(PointCloud2, '/lidar_clustering', 10)
        self.human_seg_pub = self.create_publisher(PointCloud2, '/human_seg', 10)
        self.human_dot_pub = self.create_publisher(PointCloud2, '/human_detections', 10)
        self.det_thresh = 0.5
        self.max_hum=max_hum
        
    def read_lidar_frames(self, msg):
        tf_mat = self.tfmsg_to_matrix(msg)
        if (msg.transforms[0].child_frame_id not in list(self.lidar_frames.keys())) and \
            (msg.transforms[0].child_frame_id in self.lidar_list):
            self.lidar_frames[msg.transforms[0].child_frame_id] = tf_mat
        if len(list(self.lidar_frames.keys())) == len(self.lidar_list):
            self.tf_frames_event.set()
    
    def read_lidar(self, *lidar_N):
        lidar_len = len(self.lidar_list)
        pcl_time = 0
        pcl_arrays = [self.pcl2_to_numpy(lidar_N[i], self.lidar_frames[lidar_name]) \
                      for i, lidar_name in enumerate(self.lidar_list)]
        total_pcl = np.concatenate(pcl_arrays, axis=0)
        for lidar_i, lidar in enumerate(self.lidar_list):
            pcl_time += lidar_N[lidar_i].header.stamp.sec + lidar_N[lidar_i].header.stamp.nanosec * 1e-9
        if self.curr_lidar.full(): self.curr_lidar.get_nowait()
        self.curr_lidar.put({'data': total_pcl, 'time': pcl_time / len(self.lidar_list)})
    
    def tfmsg_to_matrix(self, tf_msg):            # move to utils
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

    def pcl2_to_numpy(self, msg, tf):
        pc = ros2_numpy.numpify(msg)
        points= pc['xyz']
        points = points[np.all(np.isfinite(points), axis=1)]
        points = np.dot(points, tf[:3, :3].T) + tf[:3, 3].T
        return points

    def start_detection(self):
        print('Detection Started')
        self.detection_thread.start()

    def publish_pcl_clusters(self, pcl, clusters):
        labels = np.zeros((pcl.shape[0], 1))
        pcl = np.hstack((pcl, labels)).astype('float32')
        pcl[:, 3] = -1
        for i, clstr in enumerate(clusters):
            pcl[clstr, 3] = i
        pcl = pcl[pcl[:, 3] >=0, :]
        pcl[:, 3] /= np.unique(pcl[:, 3]).shape[0]
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.fields.append(PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1))
        msg.fields.append(PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1))
        msg.fields.append(PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1))
        msg.fields.append(PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1))
        msg.point_step = 16
        msg.height = 1
        msg.width = pcl.shape[0]
        msg.row_step = msg.point_step * pcl.shape[0]
        msg.is_bigendian = False
        msg.is_dense = True
        msg.data = pcl.tobytes()
        self.pub_cls.publish(msg)        

    def publish_humans(self, human_poses, pcl_time):
        if(human_poses.shape[0] < 1):
            return
        msg = PointCloud2()
        time_msg = Time()
        time_msg.sec = int(pcl_time)
        time_msg.nanosec = int((pcl_time - time_msg.sec) * 1e9)
        msg.header.stamp = time_msg
        msg.header.frame_id = "world"
        msg.fields.append(PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1))
        msg.fields.append(PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1))
        msg.fields.append(PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1))
        msg.point_step = 12
        msg.height = 1
        msg.width = human_poses.shape[0]
        msg.row_step = msg.point_step * human_poses.shape[0]
        msg.is_bigendian = False
        msg.is_dense = True
        msg.data = human_poses.astype(np.float32).tobytes()
        self.human_dot_pub.publish(msg)        

    def publish_human_seg(self, human_points, pcl_time):
        if(human_points.shape[0] == 0):
            return
        pcl = human_points
        msg = PointCloud2()
        time_msg = Time()
        time_msg.sec = int(pcl_time)
        time_msg.nanosec = int((pcl_time - time_msg.sec) * 1e9)
        msg.header.stamp = time_msg
        msg.header.frame_id = "world"
        msg.fields.append(PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1))
        msg.fields.append(PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1))
        msg.fields.append(PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1))
        msg.fields.append(PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1))
        msg.point_step = 16
        msg.height = 1
        msg.width = pcl.shape[0]
        msg.row_step = msg.point_step * pcl.shape[0]
        msg.is_bigendian = False
        msg.is_dense = True
        msg.data = pcl.tobytes()
        self.human_seg_pub.publish(msg)        

    def get_human_poses(self, boxes, centers, pcl, yout):
        pcl, centers, yout, boxes = np.copy(pcl), np.copy(centers), np.copy(yout), np.copy(boxes)
        # merge boxes
        tree = KDTree(pcl)
        annots = np.zeros((pcl.shape[0], ))
        times = np.zeros((pcl.shape[0], ))
        for i, box in enumerate(boxes):
            box += centers[i]
            _, idxs = tree.query(box, k=1)
            idxs = idxs.reshape(-1, )
            annots[idxs] += yout[i].reshape(-1, )
            times[idxs] += 1
        # annots[times > 0] /= times[times > 0]
        pcl = np.hstack((pcl, annots.reshape(-1, 1))).astype('float32')
        # filter pcl based on probability detection
        pcl = pcl[pcl[:, 3] > self.det_thresh]
        if(pcl.shape[0] == 0):
            return np.array([]), np.array([])
        # semantic to instanse
        clustering = DBSCAN(eps=0.5, min_samples=15).fit(pcl[:, :3])
        cluster_labels = clustering.labels_
        human_ids = np.unique(cluster_labels[cluster_labels>=0])
        if(self.max_hum !=None) and (human_ids.shape[0] > self.max_hum):
            cluster_votes = Counter(cluster_labels)
            class_ids, class_votes = np.array([key for key in cluster_votes.keys() if key >= 0]), \
                np.array([cluster_votes[key] for key in cluster_votes.keys() if key >= 0])
            class_ids = class_ids[np.argsort(class_votes).reshape(-1, )][::-1]
            mask = np.in1d(cluster_labels, class_ids).reshape(cluster_labels.shape)
            cluster_labels[~mask] = -1
        human_poses = np.zeros(shape=(human_ids.shape[0], 3))
        for i, human_id in enumerate(human_ids):
            human_poses[i, :] = pcl[cluster_labels == human_id, :3].mean(axis=0)
        return human_poses, pcl[cluster_labels >= 0, :4]

    def detection_loop(self):
        print('Waiting lidar frames')
        self.tf_frames_event.wait()
        self.destroy_subscription(self.tf_sub)
        print('Frames have been recieved. Lidar topics subscribed...')
        self.sync = ApproximateTimeSynchronizer([item[1] for item in self.lidar_sub.items()], queue_size=10, slop=0.4)
        self.sync.registerCallback(self.read_lidar)
        human_poses = []
        while(rclpy.ok()):
            if not self.curr_lidar.full(): continue
            input_lidar_info = self.curr_lidar.get()
            pcl = input_lidar_info['data']
            pcl_time = input_lidar_info['time']
            pcl = pcl_voxel(pcl, voxel_size=0.2)
            if(len(human_poses) == 0):
                cluster_idxs, pytorch_tensor, center_arr = split_pcl_to_clusters(pcl, cluster_shape=2048, min_cluster_size=40, return_pcl_gpu=True)
                self.publish_pcl_clusters(pcl, cluster_idxs)
            else:
                pytorch_tensor, center_arr = focused_split_to_boxes(pcl, human_poses, cluster_shape=2048)
            center_arr = np.concatenate([center.reshape(-1, 3) for center in center_arr])
            if(pytorch_tensor == None):
                human_poses = []
                continue
            yout, _ = self.det_model(pytorch_tensor)
            yout = torch.sigmoid(yout)
            pytorch_tensor, yout = pytorch_tensor.detach().cpu().numpy(), yout.detach().cpu().numpy()
            human_poses, human_points = self.get_human_poses(pytorch_tensor, center_arr, pcl, yout)
            self.publish_humans(human_poses, pcl_time)
            self.publish_human_seg(human_points, pcl_time)


def main():
    rclpy.init()        # initialize ros2
    # model = PointNetSeg(1).to('cuda:0').eval()      # initialize model
    # model.load_state_dict(torch.load('./results/E36_v01.17.pt'))
    model = Pointet2(radius=[0.42308816, 0.87086948, 1.19641605, 1.4843377]).to('cuda:0').eval() 
    model.load_state_dict(torch.load('./results/E36_v01.17.pt'))
    model.eval()
    human = HumanDetector(['lidar_1'], model, max_hum=1)        # initialize Human detector node
    human.start_detection()      # start detection
    rclpy.spin(human)           # start ros2 node
    rclpy.shutdown()        # end ros2 session

if __name__ == "__main__":
    main()


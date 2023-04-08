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
from utils.hierarchical_splitting import split_pcl_to_clusters
from sensor_msgs.msg import PointCloud2, PointField

class HumanDetector(Node):
    def __init__(self, lidar_list):
        super().__init__('human_detector')
        self.lidar_list = lidar_list
        self.lidar_frames, self.lidar_pcl, self.lidar_times = {}, {}, {}
        self.tf_sub = self.create_subscription(TFMessage, '/tf', self.read_lidar_frames, 20)
        self.lidar_sub = {}
        for lidar in lidar_list:
            self.lidar_sub[lidar] = Subscriber(self, PointCloud2, lidar)
        self.tf_frames_event = threading.Event()
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.curr_lidar = queue.Queue(maxsize=1)
        self.pub_cls = self.create_publisher(PointCloud2, '/lidar_3', 10)

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
    
    def merge_pcls(self):
        merged_pcl = np.empty((0, 3), dtype=np.float32)
        # lidar_times = [self.lidar_queues[lidar].get() for lidar in self.lidar_list:]
        return merged_pcl

    def voxel_downsample(self, points, voxel_size):
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        voxel_centers = np.zeros_like(voxel_indices, dtype=np.float32)
        np.add.at(voxel_centers, np.transpose(voxel_indices), points)
        voxel_counts = np.zeros_like(voxel_indices, dtype=np.int32)
        np.add.at(voxel_counts, np.transpose(voxel_indices), 1)
        voxel_centers /= voxel_counts
        return voxel_centers

    def publish_pcl_clustes(self, pcl, clusters):
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

    def detection_loop(self):
        print('Waiting lidar frames')
        self.tf_frames_event.wait()
        self.destroy_subscription(self.tf_sub)
        print('Frames have been recieved. Lidar topics subscribed...')
        self.sync = ApproximateTimeSynchronizer([item[1] for item in self.lidar_sub.items()], queue_size=10, slop=0.2)
        self.sync.registerCallback(self.read_lidar)
        while(rclpy.ok()):
            if  not self.curr_lidar.full(): continue
            input_lidar_info = self.curr_lidar.get()
            pcl = input_lidar_info['data']
            pcl = pcl_voxel(pcl, voxel_size=0.15)
            cluster_idxs = split_pcl_to_clusters(pcl, cluster_shape=2048, min_cluster_size=40)
            self.publish_pcl_clustes(pcl, cluster_idxs)
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # for idxs in cluster_idxs:
            #     ax.scatter(pcl[idxs, 0], pcl[idxs, 1], pcl[idxs, 2])
            # plt.savefig('3D_scatter_plot.png')
def main():
    rclpy.init()
    human = HumanDetector(['lidar_1', 'lidar_2'])
    human.start_detection()
    rclpy.spin(human)
    rclpy.shutdown()
if __name__ == "__main__":
    main()


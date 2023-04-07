import numpy as np
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import PointCloud2
from scipy.spatial.transform import Rotation
import threading
import struct
import matplotlib.pyplot as plt
import queue
from functools import partial
from message_filters import ApproximateTimeSynchronizer, Subscriber
from time import time
import ctypes
import ros2_numpy

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

    def read_lidar_frames(self, msg):
        tf_mat = self.tfmsg_to_matrix(msg)
        if msg.transforms[0].child_frame_id not in list(self.lidar_frames.keys()):
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
            fig,ax = plt.subplots()
            ax.scatter(pcl[:, 0], pcl[:, 1])
            fig.savefig('scatter_plot.png')


def main():
    rclpy.init()
    human = HumanDetector(['lidar_1', 'lidar_2'])
    human.start_detection()
    rclpy.spin(human)
    rclpy.shutdown()

if __name__ == "__main__":
    main()


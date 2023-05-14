import numpy as np
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import PointCloud2
import queue
from message_filters import ApproximateTimeSynchronizer, Subscriber
from time import time
from utils.o3d_funcs import pcl_voxel
from sensor_msgs.msg import PointCloud2
from models.Pointnet import PointNetSeg
from models.Pointnet2 import Pointet2
import torch
from builtin_interfaces.msg import Time
import yaml
from utils.human_detector_utils import tfmsg_to_matrix, pcl2_to_numpy


class HumanDetector(Node):
    def __init__(self, lidar_list, det_model, args):
        super().__init__('human_detector')
        self.lidar_list = lidar_list
        self.max_hum = args['max_hum']
        self.lidar_frames, self.lidar_pcl, self.lidar_times = {}, {}, {}
        self.det_model = det_model
        self.tf_sub = self.create_subscription(TFMessage, '/tf', self.read_lidar_frames, 20)
        self.hum_seg_sub = self.create_subscription(PointCloud2, '/human_seg', self.get_human_seg, 10)
        self.lidar_sub = {}
        for lidar in lidar_list:
            self.lidar_sub[lidar] = Subscriber(self, PointCloud2, lidar)
        self.curr_lidar = queue.Queue(maxsize=50)
        
    def read_lidar_frames(self, msg):
        """
        Read the LiDAR frames in the beginning only.
        """
        tf_mat = tfmsg_to_matrix(msg)
        if (msg.transforms[0].child_frame_id not in list(self.lidar_frames.keys())) and \
            (msg.transforms[0].child_frame_id in self.lidar_list):
            self.lidar_frames[msg.transforms[0].child_frame_id] = tf_mat
        if len(list(self.lidar_frames.keys())) == len(self.lidar_list):
            self.destroy_subscription(self.tf_sub)
            self.sync = ApproximateTimeSynchronizer([item[1] for item in self.lidar_sub.items()], queue_size=10, slop=0.4)
            self.sync.registerCallback(self.read_lidar)
    
    def read_lidar(self, *lidar_N):
        """
        Reads all the lidar pcls and transform them to world frame.
        """
        pcl_time = 0
        pcl_arrays = [pcl2_to_numpy(lidar_N[i], self.lidar_frames[lidar_name]) \
                      for i, lidar_name in enumerate(self.lidar_list)]
        total_pcl = np.concatenate(pcl_arrays, axis=0)
        for lidar_i, lidar in enumerate(self.lidar_list):
            pcl_time += lidar_N[lidar_i].header.stamp.sec + lidar_N[lidar_i].header.stamp.nanosec * 1e-9
        if self.curr_lidar.full(): self.curr_lidar.get_nowait()         # removes old pcls when new has come
        self.curr_lidar.put({'data': total_pcl, 'time': pcl_time / len(self.lidar_list)})
    
    def find_closest_pcl(self, det_time):
        cur_msg = []
        while(not self.curr_lidar.empty()):
            cur_msg = self.curr_lidar.get()
            if(cur_msg['time'] >= det_time):
                return cur_msg
            elif(cur_msg['time'] < det_time):
                if((not self.curr_lidar.empty()) and (abs(self.curr_lidar.queue[0]['time'] - det_time) > det_time - cur_msg['time'])):
                    return cur_msg
                else:
                    continue
        return cur_msg

    def get_human_seg(self, msg):
        det_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        human_seg = pcl2_to_numpy(msg)
        print(abs(det_time - self.find_closest_pcl(det_time)['time']))
        

def main():
    with open('config/human_det_conf.yaml', 'r') as file:
        args = yaml.safe_load(file)
    rclpy.init()        # initialize ros2
    if(args['model_name'] == 'Pointnet2'):
        model = Pointet2(radius=[0.42308816, 0.87086948, 1.19641605, 1.4843377]).to('cuda:0').eval()
        model.load_state_dict(torch.load(args['weights']))
    else:
        model = PointNetSeg(1).to('cuda:0').eval()      
        model.load_state_dict(torch.load(args['weights']))
    model.eval()
    human = HumanDetector(args['lidar_list'], model, args)
    rclpy.spin(human)           # start ros2 node
    rclpy.shutdown()        # end ros2 session

if __name__ == "__main__":
    main()


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
from scipy.spatial import KDTree
from utils.human_detector_utils import tfmsg_to_matrix, pcl2_to_numpy
from utils.o3d_funcs import pcl_gicp, pcl_voxel
from time import time


class HumanPoseEstimator(Node):
    def __init__(self, lidar_list, det_model, args):
        super().__init__('human_pose_estim')
        self.lidar_list = lidar_list
        self.max_hum = args['max_hum']
        self.lidar_frames, self.lidar_pcl, self.lidar_times = {}, {}, {}
        self.det_model = det_model
        self.files = ['showdown', 'standing', 'handsup2']
        self.poses = self.load_model(self.files)
        self.tf_sub = self.create_subscription(TFMessage, '/tf', self.read_lidar_frames, 20)
        self.det_sync = ApproximateTimeSynchronizer([Subscriber(self, PointCloud2, '/human_seg'), \
            Subscriber(self, PointCloud2, '/human_detections')], queue_size=10, slop=0.2)
        self.det_sync.registerCallback(self.find_human_pose)
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
    
    def load_model(self, files):
        poses = []
        for file in files:
            poses.append(np.fromfile(f'models/{file}.npy', dtype='float32').reshape(-1, 3))
        return poses

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

    def find_human_pose(self, human_seg, human_det):
        det_time = human_seg.header.stamp.sec + human_seg.header.stamp.nanosec * 1e-9
        human_seg = pcl2_to_numpy(human_seg)
        human_det = pcl2_to_numpy(human_det)
        matched_pcl = self.find_closest_pcl(det_time)
        if len(matched_pcl) == 0:
            return
        else:
            matched_pcl = matched_pcl['data']

        # split the segmentations to humans
        human_tree = KDTree(human_det)
        _, ii = human_tree.query(human_seg, k=1)
        ii = ii.reshape(-1, )
        human_seg = [human_seg[ii == idx] for idx in np.unique(ii)]
        
        # find the pose for each human
        for i, human_pos in enumerate(human_det.reshape(-1, 3)):
            sub_pcl = matched_pcl[np.linalg.norm(matched_pcl - human_pos.reshape(-1, 3), axis=1) < 3]
            sub_pcl_tree = KDTree(sub_pcl)
            human_seg_tree = KDTree(human_seg[i])
            idxs = sub_pcl_tree.query_ball_tree(human_seg_tree, r=0.5)
            logic_vector = np.array([True if len(idx) > 0 else False for idx in idxs])
            full_human_seg = sub_pcl[logic_vector] - human_pos
            scores = []
            for j, pose in enumerate(self.poses):
                # pose = pcl_voxel(pose, 0.1)
                gicp_res = pcl_gicp(full_human_seg, pose)
                scores.append(gicp_res.fitness)
            print(self.files[np.argmax(scores)])

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
    human = HumanPoseEstimator(args['lidar_list'], model, args)
    rclpy.spin(human)           # start ros2 node
    rclpy.shutdown()        # end ros2 session

if __name__ == "__main__":
    main()


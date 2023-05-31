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
import yaml
from scipy.spatial import KDTree
from utils.human_detector_utils import tfmsg_to_matrix, pcl2_to_numpy
from utils.o3d_funcs import pcl_gicp, pcl_voxel
from time import time
from rclpy.node import Node
from std_msgs.msg import String


class HumanPoseEstimator(Node):
    def __init__(self, lidar_list, det_model, fall_det, pose_det, args):
        super().__init__('human_pose_estim')
        self.lidar_list = lidar_list
        self.max_hum = args['max_hum']
        self.lidar_frames, self.lidar_pcl, self.lidar_times = {}, {}, {}
        self.det_model = det_model
        self.fall_det, self.pose_det = fall_det, pose_det
        self.current_pose = None
        self.last_lidar_time = 0
        self.files = ['standing', 'showdown', 'handsup2']
        self.poses = self.load_model(self.files)
        self.tf_sub = self.create_subscription(TFMessage, '/tf', self.read_lidar_frames, 20)
        self.det_sync = ApproximateTimeSynchronizer([Subscriber(self, PointCloud2, '/human_seg'), \
            Subscriber(self, PointCloud2, '/human_detections')], queue_size=10, slop=0.2)
        self.det_sync.registerCallback(self.human_det_call)
        self.lidar_sub = {}
        for lidar in lidar_list:
            self.lidar_sub[lidar] = Subscriber(self, PointCloud2, lidar)
        self.curr_lidar = queue.Queue(maxsize=100)
        self.last_pos = queue.Queue(maxsize=10)
        self.pose_publisher = self.create_publisher(String, 'human_pose', 10)
        self.timer_ = self.create_timer(0.05, self.pose_message_pub)
        if self.fall_det:
            self.missing, self.no_match, self.fall_p = False, False, False

    def read_lidar_frames(self, msg):
        """
        This is a callback function for transformation subscriber
        Reads the LiDAR frames in the beginning only once for each
        LiDAR frame and saves it in self.lidar_frames.
        :param msg: transform type message.
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
        """
        Loads the blueprint poses of intrest.
        :param files: the file names of blueprints inside the models folder.
        """
        poses = []
        for file in files:
            pose = np.fromfile(f'models/{file}.npy', dtype='float32').reshape(-1, 3)
            pose -= pose.mean(axis=0)
            poses.append(pose)
        return poses

    def read_lidar(self, *lidar_N):
        """
        This is a callback function for lidar subscriber
        Waits for N lidar to come in oder to merge and save
        them in single frame as numpy format in 
        :param *lidar_N: messages of type pointcloud queue self.curr_lidar
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
        """
        This finds the most close pointcloud to the detection time
        that human detector published.
        :param det_time: detection time
        """
        cur_msg = []
        while(not self.curr_lidar.empty()):
            cur_msg = self.curr_lidar.get()
            if(cur_msg['time'] >= det_time): return cur_msg
            elif(cur_msg['time'] < det_time):
                if((not self.curr_lidar.empty()) and (abs(self.curr_lidar.queue[0]['time'] - det_time) > det_time - cur_msg['time'])):
                    return cur_msg
                else: continue
        return cur_msg
    
    def find_human_pose(self, human_det, humans_seg, pcl):
        """
        This function finds the pose of the detected human. For now 
        it supports only one detected human. the pose is saved in variable
        self.current_pose
        :param human_det: detected human position
        :param humans_seg: the segmented points of the human
        :param pcl: the whole pcl. This is used to fill any missing points of
        the segmented human
        """
        for i, human_pos in enumerate(human_det.reshape(-1, 3)):
            # fills the segmentation of human with any missing points
            sub_pcl = pcl[np.linalg.norm(pcl - human_pos.reshape(-1, 3), axis=1) < 2.5]
            sub_pcl_tree = KDTree(sub_pcl)
            human_seg_tree = KDTree(humans_seg[i])
            idxs = sub_pcl_tree.query_ball_tree(human_seg_tree, r=0.5)
            logic_vector = np.array([True if len(idx) > 0 else False for idx in idxs])
            full_human_seg = sub_pcl[logic_vector] - human_pos
            # Search for the best matched pose
            scores = []
            tfs = []
            for j, pose in enumerate(self.poses):
                pose = pcl_voxel(pose, 0.02)
                pose[:, 2] += full_human_seg[:, 2].max() - pose[:, 2].max()     # brings the two poses in same height
                gicp_res = pcl_gicp(pose, full_human_seg, 5)
                scores.append(gicp_res.fitness)
                tfs.append(gicp_res.transformation[:3, :3])
            if(np.max(scores) < 0.25): continue
            self.current_pose = self.files[np.argmax(scores)]

    def check_human_stand(self, human_pos, human_seg, pcl):
        """
        This checks if human pose is close to a standing pose and not fall one.
        :param human_det: detected human position
        :param humans_seg: the segmented points of the human
        :param pcl: the whole pcl. This is used to fill any missing points of
        the segmented human
        """
        # fills the segmentation of human with any missing points
        sub_pcl = pcl[np.linalg.norm(pcl - human_pos.reshape(-1, 3), axis=1) < 2.5]
        sub_pcl_tree = KDTree(sub_pcl)
        human_seg_tree = KDTree(human_seg)
        idxs = sub_pcl_tree.query_ball_tree(human_seg_tree, r=0.5)
        logic_vector = np.array([True if len(idx) > 0 else False for idx in idxs])
        if(logic_vector.sum() == 0): return 0
        full_human_seg = sub_pcl[logic_vector] - human_pos
        # Search how the detected pose is look alike the standing pose
        pose = pcl_voxel(self.poses[1], 0.02)
        pose[:, 2] += full_human_seg[:, 2].max() - pose[:, 2].max()
        gicp_res = pcl_gicp(pose, full_human_seg, 2)
        return gicp_res.fitness

    def human_det_call(self, human_seg, human_det):
        """
        This is the callback that check the human body pose periodically.
        It can find its body pose or solve the similar task of fall detection.
        :param human_seg: the segmented points of the human
        :param human_det: detected human position
        the segmented human
        """
        det_time = human_seg.header.stamp.sec + human_seg.header.stamp.nanosec * 1e-9
        human_seg = pcl2_to_numpy(human_seg)
        human_det = pcl2_to_numpy(human_det)
        matched_pcl = self.find_closest_pcl(det_time)
        if len(matched_pcl) == 0:   return
        else:   matched_pcl = matched_pcl['data']
        # split the segmentations to humans
        human_tree = KDTree(human_det)
        _, ii = human_tree.query(human_seg, k=1)
        ii = ii.reshape(-1, )
        human_seg = [human_seg[ii == idx] for idx in np.unique(ii) if (ii == idx).sum() > 0]
        if(len(human_seg) == 0): return
        # find the pose for each human
        if self.pose_det:
            self.find_human_pose(human_det, human_seg, matched_pcl)
        if self.fall_det:
            # check if human has fallen
            if self.last_pos.full(): self.last_pos.get_nowait()
            self.last_pos.put({'pos': human_seg[0][:, 2].max(), 'det_time':det_time})                 # only one human 
            self.fall_p = True if human_seg[0][:, 2].max() < 1.4 else False
            self.no_match = True if self.check_human_stand(human_det[0], human_seg[0], matched_pcl) < 0.5 else False
    
    def pose_message_pub(self):
        """
        Publish the pose of the human. The final publish message is depending 
        on the application (body pose estimation or fall detection). It also checks
        when detected last time in case of fall detection.
        """
        if self.fall_det:
            if(((self.last_pos.qsize() > 0) and (self.curr_lidar.qsize() > 0)) and \
                (self.curr_lidar.queue[self.curr_lidar.qsize() - 1]['time'] - self.last_pos.queue[self.last_pos.qsize() - 1]['det_time'] > 1.5)):
                self.missing = True
            else:   self.missing = False
        msg_data = ''
        if(self.current_pose!=None):
            msg_data += f'Candidate pose {self.current_pose}'
            self.current_pose = None
        if(self.fall_det):
            if(self.missing or self.no_match or self.fall_p): 
                msg_data += f'\nFalling points:{self.fall_p}, Missing some time:{self.missing}, No standing pose:{self.no_match}'
                msg_data += '\nHuman may have fallen!'
        if(msg_data == ''): return
        msg = String()
        msg.data = msg_data
        self.pose_publisher.publish(msg)
        
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
    human = HumanPoseEstimator(args['lidar_list'], model, False, True, args)
    rclpy.spin(human)          
    rclpy.shutdown()

if __name__ == "__main__":
    main()


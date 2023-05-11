import numpy as np
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import PointCloud2
import threading
import queue
from message_filters import ApproximateTimeSynchronizer, Subscriber
from time import time
from utils.o3d_funcs import pcl_voxel
from utils.hierarchical_splitting import split_pcl_to_clusters, focused_split_to_boxes
from sensor_msgs.msg import PointCloud2, PointField
from models.Pointnet import PointNetSeg
from models.Pointnet2 import Pointet2
import torch
from builtin_interfaces.msg import Time
import yaml
from utils.human_detector_utils import merge_boxes_output, tfmsg_to_matrix, pcl2_to_numpy, semantic_to_instance, keep_k_det

class HumanDetector(Node):
    def __init__(self, lidar_list, det_model, args):
        super().__init__('human_detector')
        self.lidar_list = lidar_list
        self.batch_size = args['batch_size']
        self.det_thresh = args['det_thresh']
        self.min_human_p = args['min_human_p']
        self.min_points_hdb = args['min_points_hdb']
        self.voxel_size = args['voxel_size']
        self.max_dist_hum = args['max_dist_hum']
        self.max_hum = args['max_hum']
        self.min_hum_dist_cluster = args['min_hum_dist_cluster']
        self.hdbscan_rate = args['hdbscan_rate']
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
        
    def read_lidar_frames(self, msg):
        """
        Read the LiDAR frames in the beginning only.
        """
        tf_mat = tfmsg_to_matrix(msg)
        if (msg.transforms[0].child_frame_id not in list(self.lidar_frames.keys())) and \
            (msg.transforms[0].child_frame_id in self.lidar_list):
            self.lidar_frames[msg.transforms[0].child_frame_id] = tf_mat
        if len(list(self.lidar_frames.keys())) == len(self.lidar_list):
            self.tf_frames_event.set()
    
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

    def start_detection(self):
        print('Detection Started')
        self.detection_thread.start()

    def publish_pcl_clusters(self, pcl, clusters):
        """
        This function publish the clusters of pcl before the model inference.
        """
        if(len(clusters) == 0): return
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
        """
        Publish the mean human pose as pcl2.
        """
        if(human_poses.shape[0] < 1): return
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
        """
        Publish the human segmentation as pcl2.
        """
        if(human_points.shape[0] == 0): return
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
        pcl = merge_boxes_output(pcl, centers, yout, boxes)
        # filter pcl based on minimum probability score
        pcl = pcl[pcl[:, 3] > 0.2]
        if(pcl.shape[0] == 0): return np.array([]), np.array([])
        human_poses, cluster_labels = semantic_to_instance(pcl, self.det_thresh, self.min_human_p, self.max_dist_hum)
        human_poses, cluster_labels = keep_k_det(pcl, human_poses, cluster_labels, self.max_hum)
        return human_poses, pcl[cluster_labels >= 0, :4]

    def detection_loop(self):
        print('Waiting lidar frames')
        self.tf_frames_event.wait()
        self.destroy_subscription(self.tf_sub)
        print('Frames have been recieved. Lidar topics subscribed...')
        self.sync = ApproximateTimeSynchronizer([item[1] for item in self.lidar_sub.items()], queue_size=10, slop=0.4)
        self.sync.registerCallback(self.read_lidar)
        human_poses = []
        counter = 0
        while(rclpy.ok()):
            if not self.curr_lidar.full(): continue
            counter, pytorch_tensor, center_arr, cluster_idxs = counter + 1, torch.Tensor([]), np.zeros((0, 3)), []
            input_lidar_info = self.curr_lidar.get()
            pcl, pcl_time = input_lidar_info['data'], input_lidar_info['time']
            pcl = pcl_voxel(pcl, voxel_size=self.voxel_size)
            # Clustering
            if(len(human_poses) == 0) or np.mod(counter, self.hdbscan_rate) == 0:       # hierarchichal clustering
                cluster_idxs, pytorch_tensor, center_arr = split_pcl_to_clusters(pcl, cluster_shape=2048, min_cluster_size=self.min_points_hdb, return_pcl_gpu=True)
                if(len(center_arr) > 0): center_arr = np.concatenate([center.reshape(-1, 3) for center in center_arr], axis=0)
            if(len(human_poses) > 0):                       # human targeting clustering
                pytorch_tensor_f, center_arr_f = focused_split_to_boxes(pcl, human_poses, cluster_shape=2048, min_hum_dist=self.min_hum_dist_cluster)
                if(center_arr_f.shape[0] > 0):
                    pytorch_tensor, center_arr = torch.concatenate([pytorch_tensor, pytorch_tensor_f], dim=0), \
                        np.append(center_arr, center_arr_f, axis=0)
            if(pytorch_tensor.shape[0] == 0):continue
            # Inference
            batched_in = np.array_split(pytorch_tensor, pytorch_tensor.shape[0] // self.batch_size + 1)
            yout_tot = []
            for batch in batched_in:
                yout, _ = self.det_model(batch.to('cuda:0'))
                yout_tot.append(torch.sigmoid(yout.detach().cpu()))
            yout = torch.concatenate(yout_tot)
            pytorch_tensor, yout = pytorch_tensor.numpy(), yout.numpy()
            # publish resutls
            human_poses, human_points = self.get_human_poses(pytorch_tensor, center_arr, pcl, yout)
            self.publish_humans(human_poses, pcl_time)
            self.publish_human_seg(human_points, pcl_time)
            self.publish_pcl_clusters(pcl, cluster_idxs)

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
    human.start_detection()      # start detection
    rclpy.spin(human)           # start ros2 node
    rclpy.shutdown()        # end ros2 session

if __name__ == "__main__":
    main()


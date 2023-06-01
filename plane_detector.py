import numpy as np
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import PointCloud2
from scipy.spatial.transform import Rotation
import threading
import queue
from message_filters import ApproximateTimeSynchronizer, Subscriber
from time import time
import ros2_numpy
from utils.o3d_funcs import pcl_voxel
from utils.custom_plane_det import customDetector
from utils.plane_detector_utils import compute_convex_hull_on_plane
from scipy.spatial import Delaunay
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from utils.human_detector_utils import tfmsg_to_matrix, pcl2_to_numpy
import tf2_ros


class PlaneDetector(Node):
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
        self.detector = customDetector()
        self.plane_publisher = self.create_publisher(Marker, 'hull_marker', 10)
        
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
            self.tf_frames_event.set()
    
    def read_lidar(self, *lidar_N):
        """
        This is a callback function for lidar subscriber
        Waits for N lidar to come in oder to merge and save
        them in single frame as numpy format in 
        :param *lidar_N: messages of type pointcloud queue self.curr_lidar
        """
        lidar_len = len(self.lidar_list)
        pcl_time = 0
        pcl_arrays = [pcl2_to_numpy(lidar_N[i], self.lidar_frames[lidar_name]) \
                      for i, lidar_name in enumerate(self.lidar_list)]
        total_pcl = np.concatenate(pcl_arrays, axis=0)
        for lidar_i, lidar in enumerate(self.lidar_list):
            pcl_time += lidar_N[lidar_i].header.stamp.sec + lidar_N[lidar_i].header.stamp.nanosec * 1e-9
        if self.curr_lidar.full(): self.curr_lidar.get_nowait()
        self.curr_lidar.put({'data': total_pcl, 'time': pcl_time / len(self.lidar_list)})

    def start_detection(self):
        """
        This function starts the plane detection thread.
        """
        print('Detection Started')
        self.detection_thread.start()

    def publish_planes(self, areas_points):
        """
        This function publish the planes as Marker.TRIANGLE_LIST.
        :param areas_points: a lists of points for every plane
        """
        total_triangle_vert = np.array([]).reshape(-1, 3)
        for area_points in areas_points:
            try:
                triangulation = Delaunay(area_points)
                triangles_ind = triangulation.simplices[:, :3].reshape(-1, )
                triangle_vert = area_points[triangles_ind]
                total_triangle_vert = np.append(total_triangle_vert, triangle_vert, axis=0)
            except:
                continue
        triangles = [Point(x=vert[0], y=vert[1], z=vert[2]) for vert in total_triangle_vert]
        # Create the marker message to publish
        marker = Marker()
        marker.header.frame_id = "world"
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        marker.points = triangles
        # Publish the marker message
        self.plane_publisher.publish(marker)
    
    def detection_loop(self):
        """
        The main detection loop. It waits for the LiDAR frames to read them once. When the 
        LiDAR pcl have come, perform the custom detection pipeline.
        """
        print('Waiting lidar frames')
        self.tf_frames_event.wait()
        self.destroy_subscription(self.tf_sub)
        print('Frames have been recieved. Lidar topics subscribed...')
        self.sync = ApproximateTimeSynchronizer([item[1] for item in self.lidar_sub.items()], queue_size=10, slop=0.4)
        self.sync.registerCallback(self.read_lidar)
        while(rclpy.ok()):
            if not self.curr_lidar.full(): continue
            input_lidar_info = self.curr_lidar.get()
            pcl = input_lidar_info['data']
            pcl = pcl_voxel(pcl, voxel_size=0.3)
            planes = self.detector.detectPlanes(pcl)
            areas = compute_convex_hull_on_plane(pcl, planes)
            self.publish_planes(areas)

def main():
    rclpy.init()        # initialize ros2
    # HERE PUT THE LIDAR TOPICS you want to read
    plane_det = PlaneDetector(['lidar_1'])        # initialize Human detector node
    plane_det.start_detection()      # start detection
    rclpy.spin(plane_det)           # start ros2 node
    rclpy.shutdown()        # end ros2 session

if __name__ == "__main__":
    main()


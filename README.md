# thesis detection
## Overview
The repository below is an integral part of my academic thesis [1] at Aristotle University of Thessaloniki's Department of Electrical and Computer Engineering. The main objective of the thesis was to investigate digital indoor reconstruction and human detection algorithms using LiDAR technology. In this repository, we have created a human detector and a plane detector that operate within LiDAR point clouds. We also utilized the human detector to identify three distinct human body positions. In addition, a simple human fall detector was developed. The detectors and apps were created with the help of the intermediary software Robotic Operation System (ROS2). They can operate in real time, either in a Gazebo Simulation environment or with actual LiDAR sensors.

This repository contains, in addition to the applications, the functions used for training the human detector model, notably Pointnet and Pointnet2. The trainer, hyperparameter tuner, data processing, and visualization are among these functions. We supply the best weights for these models that produced during training in JRDB dataset. More detailed information regarding the training techniques can be obtained at the provided link [1].

## Software Requirements
Below you can see the software packages and their version used to deployment of this repository.

 - CUDA 11.7
 - Robotic Operation System (ROS2) Humble Hawksbill [6] with XACRO and Rviz2
 - Python 3.10
    - See requirements.txt and run ``pip install -r requirements.txt`` 
    - CUDF and CUML from RAPIDS [2]  https://docs.rapids.ai/install
## Seting up procedure
1. Iniside results download the weights for the models.

2. Setup the configuration in config fodler. There are two configurations one for training (``train_config.yaml``) and one for the human detector (``human_det_conf.yaml``). 

## Human Detector
The file ``human_detector.py`` contains the human detector application. Simply execute ``python3 human_detector.py``. The detector's main pipeline is as follows:
1. Reading the configuration file
2. Waiting for all LiDAR frames to be published. This is necessary in order to combine data from various LiDARs. In the first parameter list of the configuration file, the LiDAR frames can be defined.
3. Every time the LiDAR publishes a pointcloud in their ROS2 topics, their pcls are combined into a single frame.
4. Partitioning the space with HDBSCAN
5. Inference with Pointnet(++) model 
6. DBSCAN-based semantic to instance segmantation conversion
7. Publish the human poses as pcl type to the ``/human_detections`` topic.
8. Publish the semantic segmentation points to ``/human_seg`` topic as pcl type

## Human Body Pose Estimator

## Training Human Detector


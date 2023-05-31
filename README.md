# thesis detection
## Overview
The repository below is an integral part of my academic thesis [[1]](#1) at Aristotle University of Thessaloniki's Department of Electrical and Computer Engineering. The main objective of the thesis was to investigate digital indoor reconstruction and human detection algorithms using LiDAR technology. In this repository, we have created a human detector and a plane detector that operate in LiDAR point clouds. We also utilized the human detector to identify three distinct human body positions. In addition, a simple human fall detector was developed. The detectors and apps were created with the help of the intermediary software Robotic Operation System (ROS2). They can operate in real time, either in a Gazebo Simulation environment [[3]](#3) or with actual LiDAR sensors.

This repository contains, in addition to the applications, the functions used for training the human detector model, notably Pointnet and Pointnet2. The trainer, hyperparameter tuner, data processing, and visualization are among these functions. We supply the best weights for these models that produced during training in JRDB dataset. More detailed information regarding the training techniques can be obtained at the provided link [[1]](#1).
<p align="center">

 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="results/singlehumandet.gif">
   <img alt="Single Human detection" alt="drawing" width="40%" height="250">
 </picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/multihumandet.gif">
  <img alt="Multi human detection" alt="drawing" width="40%" height="250">
</picture>

 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="results/handsup_res.png">
   <img alt="Pose Estimation" alt="drawing" width="40%" height="250">
 </picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/handsup_res.png">
  <img alt="Big simulated workspace with multiple humans walking" alt="drawing" width="40%" height="250">
</picture>
</p>

## Software Requirements
Below you can see the software packages and their version used to deployment of this repository.

 - CUDA 11.7
 - Robotic Operation System (ROS2) Humble Hawksbill [[2]](#2)
 - Python 3.10
    - See requirements.txt and run ``pip install -r requirements.txt`` 
    - CUDF and CUML from RAPIDS [[4]](#4)
## Seting up procedure
1. Iniside results download the weights for the models.

2. Setup the configuration in config fodler. There are two configurations one for training (``train_config.yaml``) and one for the human detector (``human_det_conf.yaml``). 

## Human Detector
The file ``human_detector.py`` contains the human detector application. Simply execute ``python3 human_detector.py``. The detector's main pipeline is as follows:
1. Reading the configuration file
2. Waiting for all LiDAR frames to be published. This is necessary in order to combine data from various LiDARs. In the first parameter list of the configuration file, the LiDAR frames can be defined
3. Every time the LiDAR publishes a pointcloud in their ROS2 topics, their pcls are combined into a single frame
4. Partitioning the space with HDBSCAN
5. Inference with Pointnet(++) model 
6. DBSCAN-based semantic to instance segmantation conversion
7. Publish the human poses as pcl type to the ``/human_detections`` topic.
8. Publish the semantic segmentation points to ``/human_seg`` topic as pcl type

## Plane Detector
The basic detector pipeline is similar, but instead of Pointnet Models, it employs a stastical algorithm described in [[1]](#1). The pipeline is more specifically:
1. Reading the configuration with some basic parameters and the list of LiDARs
2. Waiting the LiDAR frames
3. Reading and combining income pointclouds into a single frame
4. Carry out plane detection 
5. Add the planes to the ROS2 topic ``/hull_marker`` as ``TRIANGLE_LIST`` of type ``Marker`` that can be visualized with ``RVIZ``.

## Training Human Detector
In this repository you can found a training script for Pointnet and Pointet++ models. The training configuration happens in ``train_config.yaml`` file. The training script is well documented and you can run by simply run ``python3 train.py``. More details about the strategic of training procedures can be found in thesis report [[1]](#1). The training script offers both local monitoring with outputs in console and saving of models' state in every validation step but also online monitoring with Weight and Biases. 

The process we utilized for the preprocessing of the training dataset unfolded as follows:
1. We initiated the process by reading the point clouds and their respective annotations.
2. Next, we performed space segmentation. This can happen with either overlapping boxes or HDBSCAN. In cases where the number of points was fewer than 2048, we filled the gaps with proximal points, thereby constructing overlapping spaces. 
3. Alternatively, when the count of points exceeded 2048, we reduced points with lesser partition to the cluster. Although our model is capable of handling any kind of space, maintaining 2048 points proved beneficial for batch processing of multiple spaces together and executing efficient inference.
4. Finally, we saved the subspaces and their annotations in the HDF5 format. This method ensured efficient handling of the dataset and optimized memory usage.

The described pipeline can be found inside ``create_hdf5.py`` file inside ``datasets`` folder. 

## Links and Citations
<a id="1">[1]</a> Thesis report: https://drive.google.com/file/d/1bU3LGlbmP9Ni8-itYjfeBEJv9t3pE1vR/view?usp=sharing <br>
<a id="2">[2]</a> ROS2 humble: https://docs.ros.org/en/humble/index.html <br>
<a id="3">[3]</a> Thesis Simulation: https://github.com/dimfot3/thesis_simulation <br>
<a id="4">[4]</a> CUML and CUDF: https://docs.rapids.ai/install <br>


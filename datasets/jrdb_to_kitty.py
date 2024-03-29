import argparse
import collections
import glob
import json
import multiprocessing as mp
import os
import shutil
import numpy as np
import open3d as o3d
import yaml


IN_PTC_LOWER_PATH = 'pointclouds/lower_velodyne/%s/%s.pcd'
IN_PTC_UPPER_PATH = 'pointclouds/upper_velodyne/%s/%s.pcd'
IN_LABELS_3D = 'labels_3d/*.json'
IN_CALIBRATION_F = 'calibration/defaults.yaml'

OUT_PTC_PATH = 'velodyne'
OUT_LABEL_PATH = 'label_2'
OUT_DETECTION_PATH = 'detection'

ENUM_OCCLUSION = {
    "Fully_visible": 0,
    "Mostly_visible": 1,
    "Severely_occluded": 2,
    "Fully_occluded": 3
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o',
                    '--output_dir',
                    default='/media/visitor1/DBStorage/Datasets/JRDB/KITTI',
                    help='location to store dataset in KITTI format')
    ap.add_argument('-i',
                    '--input_dir',
                    default='JRDB',
                    help='root directory in jrdb format')
    return ap.parse_args()

def get_file_list(input_dir, training=True):
    #in_type = "train_dataset_with_activity" if training else "test_dataset_without_labels"
    #input_dir = os.path.join(input_dir, in_type)

    def _filepath2filelist(path):
        return set(
            tuple(os.path.splitext(f)[0].split(os.sep)[-2:])
            for f in glob.glob(os.path.join(input_dir, path % ('*', '*'))))

    def _label2filelist(path, key='labels'):
        seq_dicts = []
        for json_f in glob.glob(os.path.join(input_dir, path)):
            with open(json_f) as f:
                labels = json.load(f)
            seq_name = os.path.basename(os.path.splitext(json_f)[0])
            seq_dicts.append({(seq_name, os.path.splitext(file_name)[0]): label
                              for file_name, label in labels[key].items()
                              })
        return dict(collections.ChainMap(*seq_dicts))

    lower_ptcs = _filepath2filelist(IN_PTC_LOWER_PATH)
    upper_ptcs = _filepath2filelist(IN_PTC_UPPER_PATH)
    labels_3d = _label2filelist(IN_LABELS_3D)
    filelist = set.intersection(lower_ptcs, upper_ptcs, labels_3d.keys())

    return {f: labels_3d[f]
            for f in sorted(filelist)}

def move_frame(input_dir, output_dir, calib, seq_name, file_name,
               labels_3d, file_idx):
    def _load_pointcloud(path, calib_key):
        ptc = np.asarray(o3d.io.read_point_cloud(
            os.path.join(input_dir, path % (seq_name, file_name))).points)
        ptc -= np.expand_dims(np.array(calib[calib_key]['translation']), 0)
        theta = float(calib[calib_key]['rotation'][-1])
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        ptc[:, :2] = np.squeeze(
            np.matmul(rotation_matrix, np.expand_dims(ptc[:, :2], 2)))
        return ptc

    # Copy point cloud
    lower_ptc = _load_pointcloud(IN_PTC_LOWER_PATH, 'lidar_lower_to_rgb')
    upper_ptc = _load_pointcloud(IN_PTC_UPPER_PATH, 'lidar_upper_to_rgb')
    ptc = np.vstack((upper_ptc, lower_ptc))

    # Save as .bin
    ptc = np.hstack((ptc, np.ones((ptc.shape[0], 1)))).astype(np.float32)
    filepath = os.path.join(output_dir, OUT_PTC_PATH, f'{file_idx:06d}.bin')
    with open(filepath, 'w') as f:
        ptc.tofile(f)
        label_id_3d = set(f['label_id'] for f in labels_3d)
        # Create label
        label_lines = []
        for label_id in label_id_3d:
            if not label_id.startswith('pedestrian:'):
                continue
            label_3d = [l for l in labels_3d if l['label_id'] == label_id][0]
            # rotation z from -2pi->2pi to 0->2pi
            label_3d['box']['rot_z'] = (label_3d['box']['rot_z'] + 2 * np.pi) % (2 * np.pi)
            rotation_z = (label_3d['box']['rot_z']
                          if label_3d['box']['rot_z'] < np.pi else -2 * np.pi +
                                                                   label_3d['box']['rot_z'])
            label_lines.append(
                f"{label_3d['observation_angle']} "
                f"{label_3d['box']['l']} {label_3d['box']['w']} "  
                f"{label_3d['box']['h']} {label_3d['box']['cx']} "
                f"{label_3d['box']['cy']} "
                f"{label_3d['box']['cz']} {rotation_z} {label_3d['attributes']['num_points']}\n"
            )
        label_out = os.path.join(output_dir, OUT_LABEL_PATH, f'{file_idx:06d}.txt')
        with open(label_out, 'w') as f:
            f.writelines(label_lines)

def convert_jr2kitti(input_dir, output_dir, file_list):
    os.makedirs(os.path.join(output_dir, OUT_PTC_PATH), exist_ok=True)
    os.makedirs(os.path.join(output_dir, OUT_LABEL_PATH), exist_ok=True)
    os.makedirs(os.path.join(output_dir, OUT_DETECTION_PATH), exist_ok=True)

    with open(os.path.join(output_dir, 'filelist.txt'), 'w') as f:
        keys = file_list.keys()
        f.write('\n'.join(a + ' ' + b for a, b in keys))
    pool = mp.Pool(5)
    with open(os.path.join(input_dir, IN_CALIBRATION_F)) as f:
        calib = yaml.safe_load(f)['calibrated']
    pool.starmap(
        move_frame,
        [(input_dir, output_dir, calib, seq_name, file_name,
            label_3d, idx)
            for idx, ((seq_name, file_name), label_3d) in enumerate(file_list.items())])
    shutil.copytree(os.path.join(input_dir, 'calibration'),
                    os.path.join(output_dir, 'calib'))

if __name__ == "__main__":
    args = parse_args()
    print('converting training set')
    file_list = get_file_list(args.input_dir, True)
    convert_jr2kitti(args.input_dir, args.output_dir, file_list)
    

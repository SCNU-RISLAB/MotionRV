#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Developed by: Xieyuanli Chen
# Updated by Jiadai Sun with new features:
#   Allows to calculate the IoU of points within a specified radius R

import argparse
import os
import yaml
import sys
import numpy as np
from tqdm import tqdm
from icecream import ic
# possible splits
splits = ["train", "valid", "test"]

# possible backends
backends = ["numpy", "torch"]


def get_args():
    parser = argparse.ArgumentParser("./evaluate_mos.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=False,
        default="/data3/zlf_data/data_odometry_velodyne/dataset",
        help='Dataset dir. No Default',
    )
    
    parser.add_argument(
        '--split', '-s',
        type=str,
        required=False,
        choices=["train", "valid", "test"],
        default="train",
        help='Split to evaluate on. One of ' +
        str(splits) + '. Defaults to %(default)s',
    )
    
    parser.add_argument(
        '--datacfg', '-dc',
        type=str,
        required=False,
        default="/data2/zlf/mos3d_v2/config/labels/semantic-kitti-mos.raw.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        required=False,
        default=None,
        help='Limit to the first "--limit" points of each scan. Useful for'
        ' evaluating single scan from aggregated pointcloud.'
        ' Defaults to %(default)s',
    )
    parser.add_argument(
        '--codalab',
        dest='codalab',
        type=str,
        default=None,
        help='Exports "scores.txt" to given output directory for codalab'
        'Defaults to %(default)s',
    )
    
    
    return parser



if __name__ == '__main__':
    
    parser = get_args()
    FLAGS, unparsed = parser.parse_known_args()
    
    assert(FLAGS.split in splits)     # assert split
    remove_static_frame = True
    # print summary of what we will do
    print("*" * 80)
    print("  INTERFACE:")
    print("  Data: ", FLAGS.dataset)
    print("  Split: ", FLAGS.split)
    print("  Config: ", FLAGS.datacfg)
    print("  Limit: ", FLAGS.limit)
    print("  Codalab: ", FLAGS.codalab)
    print("  remove_static_frame: ", remove_static_frame)

    print("*" * 80)

    print(f"Opening data config file {FLAGS.datacfg}")
    DATA = yaml.safe_load(open(FLAGS.datacfg, 'r'))

    # get number of interest classes, and the label mappings
    class_strings = DATA["labels"]
    class_remap = DATA["learning_map"]
    class_inv_remap = DATA["learning_map_inv"]
    class_ignore = DATA["learning_ignore"]
    nr_classes = len(class_inv_remap)

    # make lookup table for mapping
    maxkey = max(class_remap.keys())
    
    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(class_remap.keys())] = list(class_remap.values())

    # get test set
    test_sequences = DATA["split"][FLAGS.split]

    # get label paths
    label_names = []
    lidar_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        label_paths = os.path.join(FLAGS.dataset, "sequences", str(sequence), "labels")
        # populate the label names
        seq_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(label_paths)) for f in fn if ".label" in f]
        seq_label_names.sort()
        label_names.extend(seq_label_names)

        
        lidar_paths = os.path.join(FLAGS.dataset, "sequences", str(sequence), "velodyne")
        seq_lidar_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(lidar_paths)) for f in fn if ".bin" in f]
        seq_lidar_names.sort()
        assert len(seq_label_names) == len(seq_lidar_names)
        lidar_names.extend(seq_lidar_names)
    # print(label_names)

    print("labels: ", len(label_names))
    print("lidars: ", len(lidar_names))
    if FLAGS.split == "train" and remove_static_frame:
        remove_mapping_path = os.path.join(os.path.dirname(__file__), "../config/train_split_dynamic_pointnumber.txt")
        with open(remove_mapping_path) as fd:
            lines = fd.readlines()
            lines = [line.strip() for line in lines]
        pending_dict = {}
        for line in lines:
            if line != '':
                seq, fid, _ = line.split()
                if int(seq) in test_sequences:
                    if seq in pending_dict.keys():
                        if fid in pending_dict[seq]:
                            raise ValueError(f"!!!! Duplicate {fid} in seq {seq} in .txt file")
                        pending_dict[seq].append(fid)
                    else:
                        pending_dict[seq] = [fid]
        useful_scan_path = [path for path in lidar_names if path.split("/")[-3] in pending_dict.keys() \
                             and path.split("/")[-1][:-4] in pending_dict[path.split("/")[-3]]]
        useful_label_path = [path for path in label_names if path.split("/")[-3] in pending_dict.keys() \
                             and path.split("/")[-1][:-6] in pending_dict[path.split("/")[-3]]]
        print("new use frame: ", len(useful_scan_path))
        assert len(useful_label_path) == len(useful_scan_path)
        label_names = useful_label_path
        lidar_names = useful_scan_path
   

    # open each file, get the tensor, and make the iou comparison
    # for lidar_file, label_file, pred_file in zip(lidar_names[:], label_names[:], pred_names[:]):
    range1_moving = 0
    range1_static = 0
    range2_moving = 0
    range2_static = 0
    range3_moving = 0
    range3_static = 0
    for f_id in tqdm(range(len(label_names[:])), desc="Evaluating sequences:", ncols=80):
        label_file = label_names[f_id]

        
        pc_xyz = np.fromfile(lidar_names[f_id], dtype=np.float32).reshape((-1, 4))[:, :3]
        depth = np.linalg.norm(pc_xyz, 2, axis=1)
        
        # radius_mask = np.ones((pc_xyz.shape[0]), dtype=bool)
        
        radius_mask1 = np.logical_and(depth < 15, depth >= 0)
        radius_mask2 = np.logical_and(depth < 30, depth >= 15)
        radius_mask3 = np.logical_and(depth < 80, depth >= 30)

        # open label
        label = np.fromfile(label_file, dtype=np.int32)
        label = label.reshape((-1))  # reshape to vector
        label = label & 0xFFFF       # get lower half for semantics

        if FLAGS.limit is not None:
            label = label[:FLAGS.limit]  # limit to desired length
        label = remap_lut[label]         # remap to xentropy format

        label1 = label[radius_mask1]
        label2 = label[radius_mask2]
        label3 = label[radius_mask3]

        range1_moving += sum(label1 == 2)
        range1_static += sum(label1 == 1)

        range2_moving += sum(label2 == 2)
        range2_static += sum(label2 == 1)

        range3_moving += sum(label3 == 2)
        range3_static += sum(label3 == 1)

    total_moving = range1_moving + range2_moving + range3_moving
    total_static = range1_static + range2_static + range3_static

    print("0-15m moving points:", range1_moving)
    print("      static points:", range1_static)
    print("      total  points:", range1_moving + range1_static)

    print("15-30m moving points:", range2_moving)
    print("       static points:", range2_static)
    print("       total  points:", range2_moving + range2_static)

    print("30-80m moving points:", range3_moving)
    print("       static points:", range3_static)
    print("       total  points:", range3_moving + range3_static)

    
    print("total moving:", total_moving)
    print("total static:", total_static)
    print("total points:", total_moving + total_static)

    print("moving rate of total moving point in different range:")
    print("0-15m: ", range1_moving / total_moving)
    print("15-30m:", range2_moving / total_moving)
    print("30-80m:", range3_moving / total_moving)

    print("static rate of total static point in different range:")
    print("0-15m: ", range1_static / total_static)
    print("15-30m:", range2_static / total_static)
    print("30-80m:", range3_static / total_static)



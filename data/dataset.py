# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : dataset.py
# Description :preprocess data
# --------------------------------------

import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import os
import math
import numpy as np
import tensorflow as tf
from cfg.config import data_params, path_params, model_params, classes_map, anchors
from utils.process_utils import *
from data.augmentation import *

class Dataset(object):
    def __init__(self):
        self.data_path = path_params['data_path']
        self.anchors = anchors
        self.num_classes = model_params['num_classes']
        self.input_height = model_params['input_height']
        self.input_width = model_params['input_width']
        self.grid_height = model_params['grid_height']
        self.grid_width = model_params['grid_width']
        self.iou_threshold = model_params['iou_threshold']
        self.x_min = data_params['x_min']
        self.x_max = data_params['x_max']
        self.y_min = data_params['y_min']
        self.y_max = data_params['y_max']
        self.z_min = data_params['z_min']
        self.z_max = data_params['z_max']

    def load_bev_image(self, data_num):
        pcd_path = os.path.join(self.data_path, "object/training/livox", data_num+'.pcd')
        pts = self.load_pcd(pcd_path)
        roi_pts = self.filter_roi(pts)
        bev_image = self.transform_bev_image(roi_pts)
        return bev_image

    def load_bev_label(self, data_num):
        txt_path = os.path.join(self.data_path, "object/training/label", data_num + '.txt')
        label = self.load_label(txt_path)
        bev_label = self.transform_bev_label(label)
        return bev_label

    def load_pcd(self, pcd_path):
        pts = []
        f = open(pcd_path, 'r')
        data = f.readlines()
        f.close()

        line = data[9].strip('\n')
        pts_num = eval(line.split(' ')[-1])

        for line in data[11:]:
            line = line.strip('\n')
            xyzi = line.split(' ')
            x, y, z, i = [eval(i) for i in xyzi[:4]]
            pts.append([x, y, z, i])

        assert len(pts) == pts_num
        res = np.zeros((pts_num, len(pts[0])), dtype=np.float)
        for i in range(pts_num):
            res[i] = pts[i]

        return res

    def calc_xyz(self, data):
        center_x = (data[16] + data[19] + data[22] + data[25]) / 4.0
        center_y = (data[17] + data[20] + data[23] + data[26]) / 4.0
        center_z = (data[18] + data[21] + data[24] + data[27]) / 4.0
        return center_x, center_y, center_z

    def calc_hwl(self, data):
        height = (data[15] - data[27])
        width = math.sqrt(math.pow((data[17] - data[26]), 2) + math.pow((data[16] - data[25]), 2))
        length = math.sqrt(math.pow((data[17] - data[20]), 2) + math.pow((data[16] - data[19]), 2))
        return height, width, length

    def calc_yaw(self, data):
        angle = math.atan2(data[17] - data[26], data[16] - data[25])

        if (angle < -1.57):
            return angle + 3.14 * 1.5
        else:
            return angle - 1.57

    def cls_type_to_id(self, data):
        type = data[1]
        if type not in classes_map.keys():
            return -1
        return classes_map[type]

    def load_label(self, label_path):
        lines = [line.rstrip() for line in open(label_path)]
        num_obj = len(lines)

        index = 0
        true_boxes = np.zeros([num_obj, (6 + 1 + 1)], dtype=np.float32)
        for line in lines:
            data = line.split(' ')
            data[4:] = [float(t) for t in data[4:]]
            true_boxes[index, 0], true_boxes[index, 1], true_boxes[index, 2] = self.calc_xyz(data)
            true_boxes[index, 3], true_boxes[index, 4], true_boxes[index, 5] = self.calc_hwl(data)
            true_boxes[index, 6] = self.calc_yaw(data)
            true_boxes[index, 7] = self.cls_type_to_id(data)
            index += 1

        return true_boxes

    def transform_bev_label(self, true_box):
        bev_height = model_params['input_height']
        bev_width = model_params['input_width']

        range_x = data_params['x_max'] - data_params['x_min']
        range_y = data_params['y_max'] - data_params['y_min']

        # true_box: x, y, z, h, w, l, rz, class
        # bev_box: x, y, w, l, rz, class
        bev_box = np.zeros([150, 6], dtype=np.float32)

        index = 0
        boxes_num = true_box.shape[0]
        for j in range(boxes_num):
            if (true_box[j][1] > data_params['x_min']) & (true_box[j][1] < data_params['x_max']) & (
                    true_box[j][2] > data_params['y_min']) & (true_box[j][2] < data_params['y_max']):
                bev_box[index][0] = (true_box[j][1] + 0.5 * range_y) / range_y * bev_width
                bev_box[index][1] = true_box[j][0] / range_x * bev_height
                bev_box[index][2] = true_box[j][4] / range_y * bev_width
                bev_box[index][3] = true_box[j][5] / range_x * bev_height
                bev_box[index][4] = true_box[j][6]
                bev_box[index][5] = true_box[j][7]
                index = index + 1

        return bev_box

    def transform_bev_image(self, pts):
        bev_height = model_params['input_height']
        bev_width = model_params['input_width']

        range_x = data_params['x_max'] - data_params['x_min']
        range_y = data_params['y_max'] - data_params['y_min']

        # Discretize Feature Map
        point_cloud = np.copy(pts)
        point_cloud[:, 0] = np.int_(np.floor(point_cloud[:, 0] / range_x * (bev_height - 1)))
        point_cloud[:, 1] = np.int_(np.floor(point_cloud[:, 1] / range_y * (bev_width - 1)) + bev_width / 2)

        # sort-3times
        indices = np.lexsort((-point_cloud[:, 2], point_cloud[:, 1], point_cloud[:, 0]))
        point_cloud = point_cloud[indices]

        # Height Map
        height_map = np.zeros((bev_height, bev_width))

        _, indices = np.unique(point_cloud[:, 0:2], axis=0, return_index=True)
        point_cloud_frac = point_cloud[indices]

        # some important problem is image coordinate is (y,x), not (x,y)
        max_height = float(np.abs(data_params['z_max'] - data_params['z_min']))
        height_map[np.int_(point_cloud_frac[:, 0]), np.int_(point_cloud_frac[:, 1])] = point_cloud_frac[:, 2] / max_height

        # Intensity Map & DensityMap
        intensity_map = np.zeros((bev_height, bev_width))
        density_map = np.zeros((bev_height, bev_width))

        _, indices, counts = np.unique(point_cloud[:, 0:2],
                                       axis=0,
                                       return_index=True,
                                       return_counts=True)

        point_cloud_top = point_cloud[indices]
        normalized_counts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
        intensity_map[np.int_(point_cloud_top[:, 0]), np.int_(point_cloud_top[:, 1])] = point_cloud_top[:, 3]
        density_map[np.int_(point_cloud_top[:, 0]), np.int_(point_cloud_top[:, 1])] = normalized_counts

        rgb_map = np.zeros((bev_height, bev_width, 3))
        rgb_map[:, :, 0] = density_map      # r_map
        rgb_map[:, :, 1] = height_map       # g_map
        rgb_map[:, :, 2] = intensity_map    # b_map

        return rgb_map

    def filter_roi(self, pts):
        mask = np.where((pts[:, 0] >= self.x_min) & (pts[:, 0] <= self.x_max) &
                        (pts[:, 1] >= self.y_min) & (pts[:, 1] <= self.y_max) &
                        (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max))
        pts = pts[mask]

        return pts

    def preprocess_true_data(self, image, labels):
        image = np.array(image)
        image, labels = random_horizontal_flip(image, labels)

        anchor_array = np.array(anchors, dtype=np.float32)
        n_anchors = np.shape(anchor_array)[0]

        valid = (np.sum(labels, axis=-1) > 0).tolist()
        labels = labels[valid]

        y_true = np.zeros(shape=[self.grid_height, self.grid_width, n_anchors, (6 + 1 + self.num_classes)], dtype=np.float32)

        boxes_xy = labels[:, 0:2]
        boxes_wh = labels[:, 2:4]
        boxes_angle = labels[:, 4]
        true_boxes = np.concatenate([boxes_xy, boxes_wh, boxes_angle], axis=-1)

        anchors_max = anchor_array / 2.
        anchors_min = - anchor_array / 2.

        valid_mask = boxes_wh[:, 0] > 0
        wh = boxes_wh[valid_mask]

        # [N, 1, 2]
        wh = np.expand_dims(wh, -2)
        boxes_max = wh / 2.
        boxes_min = - wh / 2.

        # [N, 1, 2] & [5, 2] ==> [N, 5, 2]
        intersect_mins = np.maximum(boxes_min, anchors_min)
        intersect_maxs = np.minimum(boxes_max, anchors_max)
        # [N, 5, 2]
        intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchor_array[:, 0] * anchor_array[:, 1]
        # [N, 5]
        iou = intersect_area / (box_area + anchor_area - intersect_area + tf.keras.backend.epsilon())

        # Find best anchor for each true box [N]
        best_anchor = np.argmax(iou, axis=-1)

        for t, k in enumerate(best_anchor):
            i = int(np.floor(true_boxes[t, 0] / 32.))
            j = int(np.floor(true_boxes[t, 1] / 32.))
            c = labels[t, 5].astype('int32')
            y_true[j, i, k, 0:4] = true_boxes[t, 0:4]
            re = np.cos(true_boxes[t, 4])
            im = np.sin(true_boxes[t, 4])
            y_true[j, i, k, 4] = re
            y_true[j, i, k, 5] = im
            y_true[j, i, k, 6] = 1
            y_true[j, i, k, 7 + c] = 1

        return image, y_true

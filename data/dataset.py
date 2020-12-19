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
from cfg.config import data_params, path_params, model_params
from utils.process_utils import *
from data.augmentation import *

class Dataset(object):
    def __init__(self):
        self.data_path = path_params['data_path']
        self.anchors = model_params['anchors']
        self.num_classes = len(model_params['classes'])
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
        self.voxel_size = data_params['voxel_size']

    def load_bev_image(self, data_num):
        pcd_path = os.path.join(self.data_path, "object/training/livox", data_num+'.pcd')
        if not os.path.exists(pcd_path):
            raise KeyError("%s does not exist ... " % pcd_path)

        pts = self.load_pcd(pcd_path)
        roi_pts = self.filter_roi(pts)
        bev_image = self.transform_bev_image(roi_pts)
        return bev_image

    def load_bev_label(self, data_num):
        txt_path = os.path.join(self.data_path, "object/training/label", data_num + '.txt')
        if not os.path.exists(txt_path):
            raise KeyError("%s does not exist ... " %txt_path)

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

    def scale_to_255(self, a, min, max, dtype=np.uint8):
        return (((a - min) / float(max - min)) * 255).astype(dtype)

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
        if type not in model_params['classes']:
            return -1
        return model_params['classes'].index(type)

    def load_label(self, label_path):
        lines = [line.rstrip() for line in open(label_path)]
        label_list = []

        for line in lines:
            data = line.split(' ')
            data[4:] = [float(t) for t in data[4:]]
            type = data[1]
            if type not in model_params['classes']:
                continue
            label = np.zeros([8], dtype=np.float32)
            label[0], label[1], label[2] = self.calc_xyz(data)
            label[3], label[4], label[5] = self.calc_hwl(data)
            label[6] = self.calc_yaw(data)
            label[7] = self.cls_type_to_id(data)
            label_list.append(label)

        return np.array(label_list)

    def transform_bev_label(self, label):
        image_width = (self.y_max - self.y_min) / self.voxel_size
        image_height = (self.x_max - self.x_min) / self.voxel_size

        boxes_list = []
        boxes_num = label.shape[0]

        for i in range(boxes_num):
            center_x = (-label[i][1] / self.voxel_size).astype(np.int32) - int(np.floor(self.y_min / self.voxel_size))
            center_y = (-label[i][0] / self.voxel_size).astype(np.int32) + int(np.ceil(self.x_max / self.voxel_size))
            width = label[i][4] / self.voxel_size
            height = label[i][5] / self.voxel_size

            left = center_x - width / 2
            right = center_x + width / 2
            top = center_y - height / 2
            bottom = center_y + height / 2
            if ((left > image_width) or right < 0 or (top > image_height) or bottom < 0):
                continue
            if (left < 0):
                center_x = (0 + right) / 2
                width = 0 + right
            if (right > image_width):
                center_x = (image_width + left) / 2
                width = image_width - left
            if (top < 0):
                center_y = (0 + bottom) / 2
                height = 0 + bottom
            if (bottom > image_height):
                center_y = (top + image_height) / 2
                height = image_height - top

            box = [center_x, center_y, width, height, label[i][6], label[i][7]]
            boxes_list.append(box)

        while len(boxes_list) < 300:
            boxes_list.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        return np.array(boxes_list, dtype=np.float32)

    def transform_bev_image(self, pts):
        x_points = pts[:, 0]
        y_points = pts[:, 1]
        z_points = pts[:, 2]
        i_points = pts[:, 3]

        # convert to pixel position values
        x_img = (-y_points / self.voxel_size).astype(np.int32)  # x axis is -y in LIDAR
        y_img = (-x_points / self.voxel_size).astype(np.int32)  # y axis is -x in LIDAR

        # shift pixels to (0, 0)
        x_img -= int(np.floor(self.y_min / self.voxel_size))
        y_img += int(np.floor(self.x_max / self.voxel_size))

        # clip height value
        pixel_values = np.clip(a=z_points, a_min=self.z_min, a_max=self.z_max)

        # rescale the height values
        pixel_values = self.scale_to_255(pixel_values, min=self.z_min, max=self.z_max)

        # initalize empty array
        x_max = math.ceil((self.y_max - self.y_min) / self.voxel_size)
        y_max = math.ceil((self.x_max - self.x_min) / self.voxel_size)

        # Height Map & Intensity Map & Density Map
        height_map = np.zeros((y_max, x_max), dtype=np.float32)
        intensity_map = np.zeros((y_max, x_max), dtype=np.float32)
        density_map = np.zeros((y_max, x_max), dtype=np.float32)

        for k in range(0, len(pixel_values)):
            if pixel_values[k] > height_map[y_img[k], x_img[k]]:
                height_map[y_img[k], x_img[k]] = pixel_values[k]
            if i_points[k] > intensity_map[y_img[k], x_img[k]]:
                intensity_map[y_img[k], x_img[k]] = i_points[k]

            density_map[y_img[k], x_img[k]] += 1


        for j in range(0, y_max):
            for i in range(0, x_max):
                if density_map[j, i] > 0:
                    density_map[j, i] = np.minimum(1.0, np.log(density_map[j, i] + 1) / np.log(64))

        height_map /= 255.0
        intensity_map /= 255.0

        rgb_map = np.zeros((y_max, x_max, 3), dtype=np.float32)
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
        #image, labels = random_horizontal_flip(image, labels)


        anchor_array = np.array(model_params['anchors'], dtype=np.float32)
        n_anchors = np.shape(anchor_array)[0]

        valid = (np.sum(labels, axis=-1) > 0).tolist()
        labels = labels[valid]

        y_true = np.zeros(shape=[self.grid_height, self.grid_width, n_anchors, (6 + 1 + self.num_classes)], dtype=np.float32)

        boxes_xy = labels[:, 0:2]
        boxes_wh = labels[:, 2:4]
        boxes_angle = labels[:, 4:5]
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
            #print('|', j, i, k, c, t, true_boxes[t, 0], true_boxes[t, 1])

            y_true[j, i, k, 4] = re
            y_true[j, i, k, 5] = im
            y_true[j, i, k, 6] = 1
            y_true[j, i, k, 7 + c] = 1

        return image, y_true

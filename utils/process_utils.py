# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : process_utils.py
# Description :function
# --------------------------------------
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import random
import colorsys
import numpy as np
from cfg.config import color_map, model_params

def calc_iou_wh(box1_wh, box2_wh):
    """
    param box1_wh (list, tuple): Width and height of a box
    param box2_wh (list, tuple): Width and height of a box
    return (float): iou
    """
    min_w = min(box1_wh[0], box2_wh[0])
    min_h = min(box1_wh[1], box2_wh[1])
    area_r1 = box1_wh[0] * box1_wh[1]
    area_r2 = box2_wh[0] * box2_wh[1]
    intersect = min_w * min_h
    union = area_r1 + area_r2 - intersect
    return intersect / union

def calculate_iou(box_1, box_2):
    """
    calculate iou
    :param box_1: (x0, y0, x1, y1)
    :param box_2: (x0, y0, x1, y1)
    :return: value of iou
    """
    bboxes1 = np.transpose(box_1)
    bboxes2 = np.transpose(box_2)

    # 计算两个box的交集：交集左上角的点取两个box的max，交集右下角的点取两个box的min
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    # 计算两个box交集的wh：如果两个box没有交集，那么wh为0(按照计算方式wh为负数，跟0比较取最大值)
    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)

    # 交集面积
    intersection = int_h * int_w
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])  # bboxes1面积
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])  # bboxes2面积

    # iou=交集/并集
    iou = intersection / (vol1 + vol2 - intersection)

    return iou

def bboxes_sort(classes, scores, bboxes, top_k=100):
    index = np.argsort(-scores)
    classes = classes[index][:top_k]
    scores = scores[index][:top_k]
    bboxes = bboxes[index][:top_k]
    return classes, scores, bboxes

# 计算nms
def bboxes_nms(classes, scores, bboxes, nms_threshold=0.5):
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = calculate_iou(bboxes[i], bboxes[(i+1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]

def non_maximum_suppression(classes, scores, bboxes, iou_threshold=0.5):
    """
    calculate the non-maximum suppression to eliminate the overlapped box
    :param classes: shape is [num, 1] classes
    :param scores: shape is [num, 1] scores
    :param bboxes: shape is [num, 4] (xmin, ymin, xmax, ymax)
    :param nms_threshold: iou threshold
    :return:
    """
    scores = scores[..., np.newaxis]
    classes = classes[..., np.newaxis]
    results = np.concatenate([bboxes, scores, classes], axis=-1)
    classes_in_img = list(set(results[:, 7]))
    best_results = []

    for cls in classes_in_img:
        cls_mask = (np.array(results[:, 7], np.int32) == int(cls))
        cls_bboxes = results[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 6])
            best_result = cls_bboxes[max_ind]
            best_results.append(best_result)
            cls_bboxes = np.concatenate([cls_bboxes[:max_ind], cls_bboxes[max_ind + 1:]])
            overlap = calculate_iou(best_result[np.newaxis, :4], cls_bboxes[:, :4])

            weight = np.ones((len(overlap),), dtype=np.float32)
            iou_mask = overlap > iou_threshold
            weight[iou_mask] = 0.0

            cls_bboxes[:, 6] = cls_bboxes[:, 6] * weight
            score_mask = cls_bboxes[:, 6] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_results

# 筛选解码后的回归边界框
def postprocess(bboxes, obj_probs, class_probs, image_shape=(608,608), input_shape=(608, 608), threshold=0.5):
    # boxes shape——> [num, 6]
    bboxes = np.reshape(bboxes, [-1, 6])

    # image_height, image_width = image_shape
    # resize_ratio = min(input_shape[1] / image_width, input_shape[0] / image_height)
    #
    # dw = (input_shape[1] - resize_ratio * image_width) / 2
    # dh = (input_shape[0] - resize_ratio * image_height) / 2
    #
    # bboxes_xywh = bboxes[:, 0:4]
    # bboxes_xywh[:, 0::2] = 1.0 * (bboxes_xywh[:, 0::2] - dw) / resize_ratio
    # bboxes_xywh[:, 1::2] = 1.0 * (bboxes_xywh[:, 1::2] - dh) / resize_ratio
    # bboxes_xywh[:, 0::2] = np.clip(bboxes_xywh[:, 0::2], 0, image_width)
    # bboxes_xywh[:, 1::2] = np.clip(bboxes_xywh[:, 1::2], 0, image_height)
    # bboxes = np.concatenate([bboxes_xywh[..., 0:4], bboxes[..., 4:6]], axis=-1)

    bboxes = bboxes.astype(np.float32)

    # 置信度 * 类别条件概率 = 类别置信度scores
    obj_probs = np.reshape(obj_probs, [-1])
    class_probs = np.reshape(class_probs, [len(obj_probs), -1])
    class_max_index = np.argmax(class_probs, axis=1)
    class_probs = class_probs[np.arange(len(obj_probs)), class_max_index]
    scores = obj_probs * class_probs

    # 类别置信度scores > threshold的边界框bboxes留下
    keep_index = scores > threshold
    class_max_index = class_max_index[keep_index]
    scores = scores[keep_index]
    bboxes = bboxes[keep_index]

    # 排序取前400个
    class_max_index, scores, bboxes = bboxes_sort(class_max_index, scores, bboxes)

    # 计算nms
    bboxes = box_center_to_corner(bboxes)
    class_max_index, scores, bboxes = bboxes_nms(class_max_index, scores, bboxes)
    bboxes = box_corner_to_center(bboxes)
    # result = non_maximum_suppression(class_max_index, scores, bboxes)
    return bboxes, scores, class_max_index

def box_center_to_corner(bbox):
    """
    param bbox: cx, cy, w, l, angle, class
    return: x_min, y_min, x_max, y_max, angle, class
    """
    bbox_ = np.copy(bbox)
    cx = bbox_[:, 0]
    cy = bbox_[:, 1]
    width = bbox_[:, 2]
    height = bbox_[:, 3]
    bbox[:, 0] = cx - width / 2.0
    bbox[:, 1] = cy - height / 2.0
    bbox[:, 2] = cx + width / 2.0
    bbox[:, 3] = cy + height / 2.0
    return bbox

def box_corner_to_center(bbox):
    """
    param bbox: x_min, y_min, x_max, y_max, angle, class
    return:  cx, cy, w, l, angle, class
    """
    cx = (bbox[:, 0] + bbox[:, 2]) / 2.0
    cy = (bbox[:, 1] + bbox[:, 3]) / 2.0
    width = bbox[:, 2] - bbox[:, 0]
    height = bbox[:, 3] - bbox[:, 1]
    bbox[:, 0] = cx
    bbox[:, 1] = cy
    bbox[:, 2] = width
    bbox[:, 3] = height
    return bbox

def visualization(im, bboxes, scores, cls_inds, labels, thr=0.3):
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / float(len(labels)), 1., 1.) for x in range(len(labels))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    # draw image
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = cls_inds[i]

        thick = int((h + w) / 300)
        cv2.rectangle(imgcv, (box[0], box[1]), (box[2], box[3]), colors[cls_indx], thick)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        if box[1] < 20:
            text_loc = (box[0] + 2, box[1] + 15)
        else:
            text_loc = (box[0], box[1] - 10)
        cv2.putText(imgcv, mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * h, (255, 255, 255), thick // 3)
    cv2.imshow("test", imgcv)
    cv2.waitKey(1)

def draw_rotated_box(image, cy, cx, w, h, angle, score, class_id):
    """
    param: img(array): RGB image
    param: cy(int, float):  Here cy is cx in the image coordinate system
    param: cx(int, float):  Here cx is cy in the image coordinate system
    param: w(int, float):   box's width
    param: h(int, float):   box's height
    param: angle(float): rz
    param: class_id(tuple, list): the color of box, (R, G, B)
    """
    left = int(cy - w / 2)
    top = int(cx - h / 2)
    right = int(cx + h / 2)
    bottom = int(cy + h / 2)
    ro = np.sqrt(pow(left - cy, 2) + pow(top - cx, 2))
    a1 = np.arctan((w / 2) / (h / 2))
    a2 = -np.arctan((w / 2) / (h / 2))
    a3 = -np.pi + a1
    a4 = np.pi - a1
    rotated_p1_y = cy + int(ro * np.sin(angle + a1))
    rotated_p1_x = cx + int(ro * np.cos(angle + a1))
    rotated_p2_y = cy + int(ro * np.sin(angle + a2))
    rotated_p2_x = cx + int(ro * np.cos(angle + a2))
    rotated_p3_y = cy + int(ro * np.sin(angle + a3))
    rotated_p3_x = cx + int(ro * np.cos(angle + a3))
    rotated_p4_y = cy + int(ro * np.sin(angle + a4))
    rotated_p4_x = cx + int(ro * np.cos(angle + a4))
    center_p1p2y = int((rotated_p1_y + rotated_p2_y) * 0.5)
    center_p1p2x = int((rotated_p1_x + rotated_p2_x) * 0.5)

    class_name = model_params['classes'][class_id]
    color = color_map[class_name]
    cv2.line(image, (rotated_p1_y, rotated_p1_x), (rotated_p2_y, rotated_p2_x), color, 1)
    cv2.line(image, (rotated_p2_y, rotated_p2_x), (rotated_p3_y, rotated_p3_x), color, 1)
    cv2.line(image, (rotated_p3_y, rotated_p3_x), (rotated_p4_y, rotated_p4_x), color, 1)
    cv2.line(image, (rotated_p4_y, rotated_p4_x), (rotated_p1_y, rotated_p1_x), color, 1)
    cv2.line(image, (center_p1p2y, center_p1p2x), (cy, cx), color, 1)

    cv2.putText(image, class_name + ' : {:.2f}'.format(score), (left, top), cv2.FONT_HERSHEY_PLAIN, 0.7, color, 1, cv2.LINE_AA)

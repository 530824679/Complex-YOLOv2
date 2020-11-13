# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : configs.py
# Description :config parameters
# --------------------------------------
import os

path_params = {
    'data_path': '/home/chenwei/HDD/livox_dl/LIVOX',
    'checkpoints_dir': './checkpoints',
    'logs_dir': './logs',
    'tfrecord_dir': '/home/chenwei/HDD/livox_dl/LIVOX/tfrecord',
    'checkpoints_name': 'model.ckpt',
    'train_tfrecord_name': 'train.tfrecord',
    'test_tfrecord_name': 'test.tfrecord',
    'test_output_dir': './test'
}

data_params = {
    'x_min': 0.0,
    'x_max': 60.8,
    'y_min': -30.4,
    'y_max': 30.4,
    'z_min': -3.0,
    'z_max': 3.0,
    'voxel_size': 0.1,
}

model_params = {
    'image_height': 608,            # bev图片高度
    'image_width': 608,             # bev图片宽度
    'channels': 3,                  # 输入图片通道数
    'grid_height': 19,              # 输出特征图的网格高度
    'grid_width': 19,               # 输出特征图的网格宽度
    'anchor_num': 5,                # 每个网格负责预测的BBox个数
    # 'object_scale': 1.0,          # 置信度有目标权重
    # 'noobject_scale': 0.5,        # 置信度无目标权重
    # 'class_scale': 2.0,           # 分类损失权重
    # 'coord_scale': 5.0,           # 定位损失权重
    'num_classes': 4,               # 数据集的类别个数
    'iou_threshold': 0.5,
}

solver_params = {
    'gpu': '0',                     # 使用的gpu索引
    'learning_rate': 0.0001,        # 初始学习率
    'decay_steps': 30000,           #衰变步数
    'decay_rate': 0.1,              #衰变率
    'staircase': True,
    'batch_size': 8,                # 每批次输入的数据个数
    'max_iter': 100000,             # 训练的最大迭代次数
    'save_step': 1000,              # 权重保存间隔
    'log_step': 1000,               # 日志保存间隔
    'display_step': 100,            # 显示打印间隔
    'weight_decay': 0.0001,         # 正则化系数
    'restore': False                # 支持restore
}

test_params = {
    'prob_threshold': 0.01,         # 类别置信度分数阈值
    'iou_threshold': 0.1,           # nms阈值，小于0.4被过滤掉
    'max_output_size': 10           # nms选择的边界框最大数量
}

classes_map = {'car': 0, 'bus': 1, 'truck': 1, 'pedestrains': 2}

anchors = [[0.57273, 0.677385],
           [1.87446, 2.06253],
           [3.33843, 5.47434],
           [7.88282, 3.52778],
           [9.77052, 9.16828]]
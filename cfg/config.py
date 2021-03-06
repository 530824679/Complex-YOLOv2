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
    'train_data_path': '/home/chenwei/HDD/livox_dl/LIVOX/ImageSets/Main/trainval.txt',
    'checkpoints_dir': './checkpoints',
    'logs_dir': './logs',
    'tfrecord_dir': './tfrecord',
    'checkpoints_name': 'model.ckpt',
    'train_tfrecord_name': 'train.tfrecord',
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
    'input_height': 608,            # bev图片高度
    'input_width': 608,             # bev图片宽度
    'channels': 3,                  # 输入图片通道数
    'grid_height': 19,              # 输出特征图的网格高度
    'grid_width': 19,               # 输出特征图的网格宽度
    'anchor_num': 5,                # 每个网格负责预测的BBox个数
    'anchors': [[5, 5], [7, 15], [18, 43], [20, 46], [28, 104]],
    'classes': ['car', 'bus', 'truck', 'pedestrians'],  # 类别
    'num_classes': 4,               # 数据集的类别个数
    'iou_threshold': 0.5,
}

solver_params = {
    'gpu': '0',                     # 使用的gpu索引
    'lr': 1e-3,                     # 初始学习率
    'decay_steps': 5000,            # 衰变步数
    'decay_rate': 0.95,             # 衰变率
    'staircase': True,
    'batch_size': 8,                # 每批次输入的数据个数
    'epoches': 500,              # 训练的最大迭代次数
    'save_step': 1000,              # 权重保存间隔
    'log_step': 1000,               # 日志保存间隔
    'weight_decay': 0.0001,         # 正则化系数
    'restore': True                # 支持restore
}

test_params = {
    'prob_threshold': 0.5,         # 类别置信度分数阈值
    'iou_threshold': 0.5,           # nms阈值，小于0.4被过滤掉
    'max_output_size': 10           # nms选择的边界框最大数量
}

color_map = {'car': (255, 0, 0), 'bus': (0, 255, 0), 'truck': (0, 0, 255), 'pedestrians': (128, 128, 128)}
# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : dataset_utils.py
# Description :数据集清理
# --------------------------------------

import os
import numpy as np
import tensorflow as tf
from model import network
from cfg.config import path_params, model_params
from tensorflow.python.framework import graph_util

def total_sample(file_name):
    sample_nums = 0
    for record in tf.python_io.tf_record_iterator(file_name):
        sample_nums += 1
    return  sample_nums

def create_trainval_txt(root_path):
    data_path = os.path.join(root_path, 'object/training/livox')
    trainval = os.path.join(root_path, 'ImageSets/Main/trainval.txt')

    if os.path.exists(trainval):
        os.remove(trainval)

    file_obj = open(trainval, 'w', encoding='utf-8')
    file_list = os.listdir(data_path)
    for file in file_list:
        filename = os.path.splitext(file)[0]
        file_obj.writelines(filename)
        file_obj.write('\n')
    file_obj.close()

def freeze_graph(checkpoints_path, output_graph):
    """
    :param checkpoints_path: ckpt文件路径
    :param output_graph: pb模型保存路径
    :return:
    """
    with tf.Graph().as_default():
        images = tf.placeholder(shape=[None, model_params['input_height'], model_params['input_width'], model_params['channels']], dtype=tf.float32, name='inputs')
        output_node_names = "yolo_v2/logits"
        with tf.variable_scope('yolo_v2'):
            Model = network.Network(is_train=False)
            logits = Model.build_network(images)
            with tf.Session() as sess:
                saver = tf.train.Saver()
                saver.restore(sess, checkpoints_path)  #恢复图并得到数据
                graph = tf.get_default_graph()
                output_graph_def = graph_util.convert_variables_to_constants(
                    sess=sess,
                    input_graph_def=sess.graph_def,
                    output_node_names=output_node_names.split(","))

                # remove training nodes
                output_graph_def = graph_util.remove_training_nodes(
                    output_graph_def)  #删除训练层，只保留主干
                with tf.gfile.GFile(output_graph, "wb") as f:  #保存模型
                    f.write(output_graph_def.SerializeToString())  #序列化输出
                print("%d ops in the final graph." %len(output_graph_def.node))  #得到当前图有几个操作节点


if __name__ == '__main__':
    create_trainval_txt(path_params['data_path'])
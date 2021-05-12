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
        image = tf.placeholder(shape=[None, 608, 608, 3], dtype=tf.float32, name='inputs')

        # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
        output_node_names = "reorg_layer/obj_probs,reorg_layer/class_probs,reorg_layer/bboxes_probs"

        # 从模型代码中获取结构
        Model = network.Network(is_train=False)
        logits = Model.build_network(image)
        output = Model.reorg_layer(logits, model_params['anchors'])

        # 从meta中获取结构
        #saver = tf.train.import_meta_graph(checkpoints_path + '.meta', clear_devices=True)

        # 获得默认的图
        graph = tf.get_default_graph()

        # 返回一个序列化的图代表当前的图
        input_graph_def = graph.as_graph_def()

        with tf.Session() as sess:
            saver = tf.train.Saver()
            # 恢复图并得到数据
            saver.restore(sess, checkpoints_path)

            # 模型持久化，将变量值固定
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=input_graph_def,
                output_node_names=output_node_names.split(","))

            # 删除训练层，只保留主干
            output_graph_def = graph_util.remove_training_nodes(
                output_graph_def)

            # 保存模型
            with tf.gfile.GFile(output_graph, "wb") as f:

                # 序列化输出
                f.write(output_graph_def.SerializeToString())

            # 得到当前图有几个操作节点
            print("%d ops in the final graph." %len(output_graph_def.node))

            # for op in graph.get_operations():
            #     print(op.name, op.values())

if __name__ == '__main__':
    #create_trainval_txt(path_params['data_path'])
    input_checkpoint='/home/chenwei/HDD/Project/Complex-YOLOv2/checkpoints/model.ckpt-43'
    out_pb_path="/home/chenwei/HDD/Project/Complex-YOLOv2/pb/frozen_model.pb"
    freeze_graph(input_checkpoint, out_pb_path)
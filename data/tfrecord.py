# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : tfrecord.py
# Description :create and parse tfrecord
# --------------------------------------

import os
import numpy as np
import tensorflow as tf
from data.dataset import Dataset
from cfg.config import path_params, model_params, solver_params, classes_map

class TFRecord(object):
    def __init__(self):
        self.data_path = path_params['data_path']
        self.tfrecord_dir = path_params['tfrecord_dir']
        self.train_tfrecord_name = path_params['train_tfrecord_name']
        self.input_width = model_params['input_width']
        self.input_height = model_params['input_height']
        self.channels = model_params['channels']
        self.grid_height = model_params['grid_height']
        self.grid_width = model_params['grid_width']
        self.class_num = model_params['num_classes']
        self.batch_size = solver_params['batch_size']
        self.dataset = Dataset()

    # 数值形式的数据,首先转换为string,再转换为int形式进行保存
    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    # 数组形式的数据,首先转换为string,再转换为二进制形式进行保存
    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def create_tfrecord(self):
        # 获取作为训练验证集的图片序列
        trainval_path = os.path.join(self.data_path, 'ImageSets', 'Main', 'trainval.txt')

        tf_file = os.path.join(self.tfrecord_dir, self.train_tfrecord_name)
        if os.path.exists(tf_file):
            os.remove(tf_file)

        writer = tf.python_io.TFRecordWriter(tf_file)
        with open(trainval_path, 'r') as read:
            lines = read.readlines()
            for line in lines:
                pcd_num = line[0:-1]
                image = self.dataset.load_bev_image(pcd_num)
                bboxes = self.dataset.load_bev_label(pcd_num)

                image_raw = image.tobytes()
                bbox_raw = bboxes.tobytes()

                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                        'bbox': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox_raw]))
                    }))
                writer.write(example.SerializeToString())
        writer.close()
        print('Finish trainval.tfrecord Done')

    def parse_single_example(self, serialized_example):
        """
        :param file_name:待解析的tfrecord文件的名称
        :return: 从文件中解析出的单个样本的相关特征，image, label
        """
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'bbox': tf.FixedLenFeature([], tf.string)
            })

        tf_image = tf.decode_raw(features['image'], tf.uint8)
        tf_bbox = tf.decode_raw(features['bbox'], tf.float32)

        tf_image = tf.reshape(tf_image, [self.input_height, self.input_width, 3])
        tf_label = tf.reshape(tf_bbox, [150, 7])

        tf_image, y_true = tf.py_func(self.dataset.preprocess_true_data, inp=[tf_image, tf_label], Tout=[tf.float32, tf.float32])
        y_true = tf.reshape(y_true, [self.grid_height, self.grid_width, 5, 6])

        return tf_image, y_true

    def create_dataset(self, filenames, batch_num, batch_size=1, is_shuffle=False):
        """
        :param filenames: record file names
        :param batch_size: batch size
        :param is_shuffle: whether shuffle
        :param n_repeats: number of repeats
        :return:
        """
        dataset = tf.data.TFRecordDataset(filenames)

        dataset = dataset.map(self.parse_single_example, num_parallel_calls=4)
        if is_shuffle:
            dataset = dataset.shuffle(batch_num)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(batch_size)

        return dataset

if __name__ == '__main__':
    tfrecord = TFRecord()
    tfrecord.create_tfrecord()

    # file = './tfrecord/train.tfrecord'
    # tfrecord = TFRecord()
    # batch_example, batch_label = tfrecord.parse_batch_examples(file)
    # with tf.Session() as sess:
    #
    #     init_op = tf.global_variables_initializer()
    #     sess.run(init_op)
    #
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     for i in range(1):
    #         example, label = sess.run([batch_example, batch_label])
    #         print(label)
    #         print(label.astype(np.float32))
    #         box = label[0, ]
    #         # cv2.imshow('w', example[0, :, :, :])
    #         # cv2.waitKey(0)
    #         print(np.shape(example), np.shape(label))
    #     # cv2.imshow('img', example)
    #     # cv2.waitKey(0)
    #     # print(type(example))
    #     coord.request_stop()
    #     # coord.clear_stop()
    #     coord.join(threads)
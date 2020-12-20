# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : tfrecord.py
# Description :create and parse tfrecord
# --------------------------------------
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import os
import numpy as np
import tensorflow as tf
from data.dataset import Dataset
from cfg.config import path_params, model_params, solver_params

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
        self.anchor_num = model_params['anchor_num']
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
                print(pcd_num)
                image = self.dataset.load_bev_image(pcd_num)
                bboxes = self.dataset.load_bev_label(pcd_num)

                valid = (np.sum(bboxes, axis=-1) > 0).tolist()
                boxes = bboxes[valid].tolist()
                if len(boxes) == 0:
                    continue

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

        tf_image = tf.decode_raw(features['image'], tf.float32)
        tf_bbox = tf.decode_raw(features['bbox'], tf.float32)

        tf_image = tf.reshape(tf_image, [self.input_height, self.input_width, 3])
        tf_label = tf.reshape(tf_bbox, [300, 6])

        tf_image, y_true = tf.py_func(self.dataset.preprocess_true_data, inp=[tf_image, tf_label], Tout=[tf.float32, tf.float32])
        y_true = tf.reshape(y_true, [self.grid_height, self.grid_width, 5, 6 + 1 + self.class_num])

        return tf_image, y_true

    def create_dataset(self, filenames, batch_num, batch_size=1, is_shuffle=False):
        """
        :param filenames: record file names
        :param batch_size: batch size
        :param is_shuffle: whether shuffle
        :param n_repeats: number of repeats
        :return:
        """
        dataset = tf.data.TextLineDataset(filenames)
        if is_shuffle:
            dataset = dataset.shuffle(batch_num)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(lambda x: tf.py_func(self.get_data, inp=[x], Tout=[tf.float32, tf.float32]), num_parallel_calls=8)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(batch_size)

        return dataset

    def process_data(self, line):
        if 'str' not in str(type(line)):
            line = line.decode()
        data = line.split()
        pcd_num = data[0]
        image = self.dataset.load_bev_image(pcd_num)
        bboxes = self.dataset.load_bev_label(pcd_num)

        image, bboxes = self.dataset.preprocess_true_data(image, bboxes)

        return image, bboxes

    def get_data(self, batch_lines):
        batch_image = np.zeros((solver_params['batch_size'], model_params['input_height'], model_params['input_width'], 3), dtype=np.float32)
        batch_label = np.zeros((solver_params['batch_size'], self.grid_height, self.grid_width, self.anchor_num, (6 + 1 + self.class_num)), dtype=np.float32)

        for num, line in enumerate(batch_lines):
            image, label = self.process_data(line)
            batch_image[num, :, :, :] = image
            batch_label[num, :, :, :, :] = label

        return batch_image, batch_label

if __name__ == '__main__':
    tfrecord = TFRecord()
    tfrecord.create_tfrecord()

    # import matplotlib.pyplot as plt
    # file = '/home/chenwei/HDD/Project/Complex-YOLOv2/tfrecord/train.tfrecord'
    # tfrecord = TFRecord()
    # dataset = tfrecord.create_dataset(file, batch_num=1, batch_size=1, is_shuffle=False)
    # iterator = dataset.make_one_shot_iterator()
    # images, labels = iterator.get_next()
    #
    # with tf.Session() as sess:
    #     for i in range(20):
    #         images_, labels_ = sess.run([images, labels])
    #         print(images_.shape, labels.shape)
    #         for images_i, boxes_i in zip(images_, labels_):
    #             data = Dataset()
    #             image_rgb = cv2.cvtColor(images_i, cv2.COLOR_RGB2BGR)
    #             valid = (np.sum(boxes_i, axis=-1) > 0).tolist()
    #             boxes = boxes_i[valid].tolist()
    #             for box in boxes:
    #                 cv2.rectangle(image_rgb, (int(box[0] - box[2] / 2), int(box[1] - box[3] / 2)), (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2)), (255, 0, 0), 2)
    #             cv2.imshow("image", image_rgb)
    #             cv2.waitKey(0)
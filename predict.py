import os
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import numpy as np
import tensorflow as tf
from cfg.config import path_params, model_params
from model import network
from data.dataset import Dataset
from utils.process_utils import *

def predict(test_dir, checkpoints):
    """
    本函数用于对测试
    :param test_dir:待测试的目录
    :param checkpoints:权重文件
    :return:
    """
    input = tf.placeholder(tf.float32, [None, model_params['image_size'], model_params['image_size'], model_params['channels']], name='input')

    # 构建网络
    model = network.Network(input)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoints)

        file_list = os.listdir(test_dir)
        for filename in file_list:
            file = os.path.join(test_dir, filename)

            image = cv2.imread(file)
            image_width = np.shape(image)[0]
            image_height = np.shape(image)[1]

            dataset = Dataset()
            image_cp = dataset.load_image((image, model_params['image_size']))

            bboxes, obj_probs, class_probs = sess.run(model.output, feed_dict={input: image})
            bboxes, scores, class_max_index = postprocess(bboxes, obj_probs, class_probs, model_params['image_size'])

            visualization(image, bboxes, scores, class_max_index, class_names)

if __name__ == '__main__':
    test_dir = path_params['test_output_dir']
    checkpoints_dir = path_params['checkpoints_dir']
    checkpoints_name = path_params['checkpoints_name']

    checkpoints_file = os.path.join(checkpoints_dir, checkpoints_name)
    predict(test_dir, checkpoints_file)
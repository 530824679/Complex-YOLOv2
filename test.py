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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test(test_path):
    """
    本函数用于对测试
    :param test_path:待测试的目录
    :return:
    """
    input = tf.placeholder(tf.float32, [None, model_params['input_height'], model_params['input_width'], model_params['channels']], name='input')

    # 构建网络
    Model = network.Network(is_train=False)
    logits = Model.build_network(input)
    output = Model.reorg_layer(logits, model_params['anchors'])

    data = Dataset()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./checkpoints/model.ckpt-14")

        file_list = os.listdir(test_path)
        for filename in file_list:
            file = os.path.join(test_path, filename)

            pts = data.load_pcd(file)
            roi_pts = data.filter_roi(pts)
            bev_image = data.transform_bev_image(roi_pts)
            bev_data = bev_image[np.newaxis, ...]

            bboxes, obj_probs, class_probs = sess.run(output, feed_dict={input: bev_data})
            bboxes, scores, class_max_index = postprocess(bboxes, obj_probs, class_probs)
            # results = postprocess(bboxes, obj_probs, class_probs)
            # for result in results:
            #     angle = data.calc_angle(result[5], result[4])
            #     draw_rotated_box(bev_image, int(result[0]), int(result[1]), int(result[2]), int(result[3]), angle, result[6], int(result[7]))

            for bbox, score, class_index in zip(bboxes, scores, class_max_index):
                angle = data.calc_angle(bbox[5], bbox[4])
                draw_rotated_box(bev_image, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), angle, score, class_index)
            cv2.imshow("image", bev_image)
            cv2.waitKey(0)
        #visualization(bev_image, bboxes, scores, class_max_index, model_params['classes'])

if __name__ == '__main__':
    save_path = path_params['test_output_dir']
    test(save_path)
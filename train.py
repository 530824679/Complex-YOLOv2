# -*- coding: utf-8 -*-

import os
import time
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import tqdm

from cfg.config import path_params, model_params, solver_params
from model import network
from data import dataset, tfrecord
from utils.dataset_utils import *

def train():
    start_step = 0
    restore = solver_params['restore']
    checkpoint_dir = path_params['checkpoints_dir']
    checkpoints_name = path_params['checkpoints_name']
    tfrecord_dir = path_params['tfrecord_dir']
    tfrecord_name = path_params['train_tfrecord_name']
    log_dir = path_params['logs_dir']
    batch_size = solver_params['batch_size']
    dataset_path = path_params['train_data_path']

    # 配置GPU
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    # 解析得到训练样本以及标注
    data_num = len(open(dataset_path, 'r').readlines())
    batch_num = int(math.ceil(float(data_num) / batch_size))
    data = tfrecord.TFRecord()
    dataset = data.create_dataset(dataset_path, batch_num, batch_size=batch_size, is_shuffle=True)
    iterator = dataset.make_one_shot_iterator()
    images, y_true = iterator.get_next()

    images.set_shape([None, 608, 608, 3])
    y_true.set_shape([None, 19, 19, 5, 7 + model_params['num_classes']])

    # 构建网络
    Model = network.Network(is_train=True)
    logits = Model.build_network(images)

    # 计算损失函数
    total_loss, diou_loss, angle_loss, confs_loss, class_loss = Model.calc_loss(logits, y_true)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(solver_params['lr'], global_step, solver_params['decay_steps'],
                                               solver_params['decay_rate'], staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss, global_step=global_step)

    # 配置tensorboard
    tf.summary.scalar("learning_rate", learning_rate)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar("diou_loss", diou_loss)
    tf.summary.scalar("angle_loss", angle_loss)
    tf.summary.scalar("confs_loss", confs_loss)
    tf.summary.scalar("class_loss", class_loss)

    # 配置tensorboard
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph(), flush_secs=60)

    # 模型保存
    save_variable = tf.global_variables()
    saver = tf.train.Saver(save_variable, max_to_keep=50)
    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        if restore == True:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                stem = os.path.basename(ckpt.model_checkpoint_path)
                restore_step = int(stem.split('.')[1].split('-')[-1])
                start_step = restore_step
                sess.run(global_step.assign(restore_step))
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Restoreing from {}'.format(ckpt.model_checkpoint_path))
            else:
                print("Failed to find a checkpoint")

        summary_writer.add_graph(sess.graph)

        import time
        print('\n----------- start to train -----------\n')
        for epoch in range(start_step + 1, solver_params['epoches']):
            train_epoch_loss, train_epoch_diou_loss, train_epoch_angle_loss, train_epoch_confs_loss, train_epoch_class_loss = [], [], [], [], []
            for index in tqdm(range(batch_num)):
                start = time.time()
                _, summary_, loss_, diou_loss_, angle_loss_, confs_loss_, class_loss_, global_step_, lr = sess.run(
                    [train_op, summary_op, total_loss, diou_loss, angle_loss, confs_loss, class_loss, global_step, learning_rate])

                print("Epoch: {}, global_step: {}, lr: {:.8f}, total_loss: {:.3f}, diou_loss: {:.3f}, angle_loss: {:.3f},confs_loss: {:.3f}, class_loss: {:.3f}".format(
                        epoch, global_step_, lr, loss_, diou_loss_, angle_loss_, confs_loss_, class_loss_))
                print("train time:", time.time() - start)
                train_epoch_loss.append(loss_)
                train_epoch_diou_loss.append(diou_loss_)
                train_epoch_angle_loss.append(angle_loss_)
                train_epoch_confs_loss.append(confs_loss_)
                train_epoch_class_loss.append(class_loss_)

                summary_writer.add_summary(summary_, global_step_)

            train_epoch_loss, train_epoch_diou_loss, train_epoch_angle_loss, train_epoch_confs_loss, train_epoch_class_loss = np.mean(
                train_epoch_loss), np.mean(train_epoch_diou_loss), np.mean(train_epoch_angle_loss),np.mean(train_epoch_confs_loss), np.mean(train_epoch_class_loss)
            print("Epoch: {}, global_step: {}, lr: {:.8f}, total_loss: {:.3f}, diou_loss: {:.3f}, angle_loss: {:.3f},confs_loss: {:.3f}, class_loss: {:.3f}".format(
                epoch, global_step_, lr, train_epoch_loss, train_epoch_diou_loss, train_epoch_angle_loss, train_epoch_confs_loss, train_epoch_class_loss))
            saver.save(sess, os.path.join(checkpoint_dir, checkpoints_name), global_step=epoch)

        sess.close()

if __name__ == '__main__':
    train()
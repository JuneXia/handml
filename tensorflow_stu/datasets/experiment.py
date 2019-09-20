# -*- coding: UTF-8 -*-
import os
import sys
project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_path)

from datasets import dataset as datset
import tensorflow as tf
import math

import socket
import getpass


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
home_path = os.environ['HOME']

user_name = getpass.getuser()
host_name = socket.gethostname()

if user_name in ['xiajun', 'yp']:
    g_datapath = os.path.join(home_path, 'res/mnist/train')
elif user_name == 'xiaj':
    g_datapath = os.path.join(home_path, 'res/mnist')
else:
    print('unkown user_name:{}'.format(user_name))
    exit(0)

tf.enable_eager_execution()
print('is eager executing: ', tf.executing_eagerly())


if __name__ == '__main__':
    validation_images_path, validation_images_label = datset.load_dataset(g_datapath)

    epoch_size = 2
    batch_size = 100
    buffer_size = 1000
    repeat = 1
    num_batch = math.ceil(len(validation_images_path) / batch_size)  # 迭代一轮所需要的训练次数

    filenames = tf.constant(validation_images_path)
    labels = tf.constant(validation_images_label)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(datset._parse_function)
    dataset = dataset.shuffle(buffer_size=buffer_size, seed=tf.set_random_seed(666),
                              reshuffle_each_iteration=True).batch(batch_size)

    if not tf.executing_eagerly():  # 非eager模式
        if False:  # 一次性迭代器
            iterator = dataset.make_one_shot_iterator()
            dataset = iterator.get_next()

            with tf.Session() as sess:
                for j in range(num_batch):
                    data = sess.run(dataset)
                    print(j, data[0].shape, data[1].shape)
        elif True:  # 多次迭代器
            dataset = dataset.repeat(epoch_size)
            iterator = dataset.make_initializable_iterator()
            dataset = iterator.get_next()
            with tf.Session() as sess:
                sess.run(iterator.initializer)

                # 消耗迭代器数据方式1：
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                while True:
                    try:
                        images, labels = sess.run(dataset)
                        print('images.shape={}, labels.shape={}'.format(images.shape, labels.shape))
                    except tf.errors.OutOfRangeError as e:
                        # 迭代完后，如果还想要继续从头迭代，可以再次sess.run(iterator.initializer)即可。
                        print(e)
                        break
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # 消耗迭代器数据方式2：
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                for epoch in range(epoch_size):
                    for j in range(num_batch):
                        data = sess.run(dataset)
                        print(j, data[0].shape, data[1].shape)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:  # eager模式
        for (batch, (images, labels)) in enumerate(dataset):
            print(batch, images.shape, labels.shape)

        if False:
            # 在eager模式下，如果使用one shot迭代器的话，并且用下面的方法消耗迭代器数据的话，则会无限迭代下去
            # 所以：为了规范起见，在eager模式下最好不要下面这种方式取迭代。
            iterator = dataset.make_one_shot_iterator()
            dataset = iterator.get_next()
            for i in range(100000):
                data = dataset
                print(i, data[0].shape, data[1].shape)

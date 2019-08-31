"""
说明：本篇代码是使用tf.data.Dataset加载数据，然后使用model.fit训练
"""

import os
import sys
project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_path)

from network import lenet3 as net
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from utils import util
import socket
import getpass


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
home_path = os.environ['HOME']

user_name = getpass.getuser()
host_name = socket.gethostname()

if user_name == 'xiajun':
    g_datapath = os.path.join(home_path, 'res/mnist/train')
elif user_name == 'xiaj':
    g_datapath = os.path.join(home_path, 'res/mnist')
else:
    print('unkown user_name:{}'.format(user_name))
    exit(0)

# Change Mark: 相对于lenet2.py，本次实验必须使用eager模式，否则会报错
tf.enable_eager_execution()
print('is eager executing: ', tf.executing_eagerly())


def _parse_function(filename, label):
    imsize = (28, 28)
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, imsize)
    image = np.expand_dims(image_resized, axis=3) / 255
    return image[0], label


def sequence_dataset(datas_path, labels, num_class, batch_size=1, buffer_size=1000, one_hot=True):
    if one_hot:
        labels = to_categorical(labels, num_class).astype(np.float)

    filenames = tf.constant(datas_path)
    labels = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=buffer_size, seed=666, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()  # dataset.repeat()不传参数表示迭代无数轮。

    # iterator = dataset.make_initializable_iterator()
    # next_element = iterator.get_next()

    return dataset


if __name__ == '__main__':
    images_path, images_label = util.get_dataset(g_datapath)
    num_class = len(set(images_label))
    batch_size = 100
    dataset = sequence_dataset(images_path, images_label, num_class, batch_size=batch_size)

    '''
    for data in dataset:
        print(data)
    '''

    model = tf.keras.Sequential([
        net.LeNet(),
        layers.Dense(num_class, activation='softmax')
    ])

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(dataset, epochs=10, steps_per_epoch=len(images_path) // batch_size + 1)


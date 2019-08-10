import os
import sys

from utils import util

sys.path.append('/home/xiaj/dev/handml/tensorflow_stu')
from network import lenet3 as net
from datasets import dataset

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


from tensorflow.examples.tutorials import mnist


# Change Mark: 相对于lenet2.py，本次实验必须使用eager模式，否则会报错
# tf.enable_eager_execution()
# print('is eager executing: ', tf.executing_eagerly())


if __name__ == '__main__1':
    (mnist_images, mnist_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    mnist_labels = to_categorical(mnist_labels, 10)
    mnist_images = np.expand_dims(mnist_images, axis=3) / 255
    mnist_labels = mnist_labels.astype(np.float)

    model = tf.keras.Sequential([
        net.LeNet(),

        # 这种先用sigmoid再用softmax的方法也可以，但是实测准确率没有直接用softmax高
        # ****************************************
        # layers.Dense(10, activation='sigmoid')
        # layers.Activation('softmax')
        # ****************************************

        # OK, GOOD
        # ****************************************
        layers.Dense(10, activation='softmax')
        # ****************************************
    ])

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(mnist_images, mnist_labels, batch_size=32, epochs=5)


class ClassificationNet(object):
    def __init__(self, input_hold, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()

        self.prelogits = embedding_net(input_hold)

        weight_decay = 5e-4
        self.prelogits = self.prelogits
        self.n_classes = n_classes
        self.embeddings = tf.nn.l2_normalize(self.prelogits, 1, 1e-10, name='embeddings')

        self.logits = slim.fully_connected(self.prelogits, n_classes, activation_fn=None,
                                      weights_initializer=slim.initializers.xavier_initializer(),
                                      weights_regularizer=slim.l2_regularizer(weight_decay),
                                      scope='Logits', reuse=False)


def _parse_function(filename, label):
    imsize = (28, 28)
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, imsize)
    image = np.expand_dims(image_resized, axis=3) / 255
    return image[0], label


def sequence_dataset(images_path, images_label, num_class, batch_size=1, one_hot=True):
    if one_hot:
        images_label = to_categorical(images_label, num_class).astype(np.float)

    filenames = tf.constant(images_path)
    labels = tf.constant(images_label)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=10000, seed=666, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()  # dataset.repeat()不传参数表示迭代无数轮。

    # iterator = dataset.make_initializable_iterator()
    # next_element = iterator.get_next()

    return dataset


if __name__ == '__main__2':
    data_path = '/home/xiaj/res/mnist'
    images_path, images_label = util.get_dataset(data_path)
    num_class = len(set(images_label))
    batch_size = 100
    dataset = sequence_dataset(images_path, images_label, num_class, batch_size=batch_size)

    model = tf.keras.Sequential([
        net.LeNet(),
        layers.Dense(num_class, activation='softmax')
    ])

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(dataset, epochs=10, steps_per_epoch=len(images_path)//batch_size+1)

>>> Siamese Triplet 训练
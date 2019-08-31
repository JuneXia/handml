import os
import sys
project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_path)

from datasets import dataset as datset
import tensorflow as tf
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


def loss(model, x, y):
    """ tensorflow 1.12及其以下版本仅仅支持这些相对较低级的API。
    目前实测代码能够跑通，但效果不佳。
    :param model:
    :param x:
    :param y:
    :return:
    """
    y_ = model(x)
    loss_val = tf.keras.losses.categorical_crossentropy(y, y_)
    train_acc = tf.keras.metrics.categorical_accuracy(y, y_)
    print(train_acc)
    return loss_val


def train_step(model, images, labels):
    """ tensorflow 1.12及其以下版本仅仅支持这些相对较低级的API。
    目前实测代码能够跑通，但效果不佳。
    :param model:
    :param images:
    :param labels:
    :return:
    """
    with tf.GradientTape() as t:
        pred = model(images)
        loss_step = tf.keras.losses.categorical_crossentropy(labels, pred)
    grads = t.gradient(loss_step, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss = loss_step.numpy().mean()

    train_acc = tf.keras.metrics.categorical_accuracy(labels, pred)
    train_acc = train_acc.numpy().mean()
    print(train_loss, train_acc)


if __name__ == '__main__':
    images_path, images_label = util.get_dataset(g_datapath)
    num_class = len(set(images_label))
    batch_size = 100

    dataset = datset.Dataset(images_path, images_label, shape=(28, 28, 1), batch_size=batch_size)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, [3, 3], activation='relu', input_shape=(None, None, 1)),
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.GlobalMaxPool2D(),
        tf.keras.layers.Dense(10)
    ])

    # optimizer = tf.keras.optimizers.Adam()
    optimizer = tf.train.AdamOptimizer()

    for epoch in range(10):
        for batch, (images2, labels) in enumerate(dataset):
            train_step(model, images2, labels)


    print('debug')


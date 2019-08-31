"""
说明：本篇代码是先把数据加载到内存，然后使用model.fit训练
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


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Change Mark: 相对于lenet2.py，本次实验必须使用eager模式，否则会报错
tf.enable_eager_execution()
print('is eager executing: ', tf.executing_eagerly())


if __name__ == '__main__':
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


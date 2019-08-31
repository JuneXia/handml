"""
说明：使用继承自keras.utils.Sequence的DataGenerator生成数据，然后使用model.fit_generator训练，
这可以自定义产生数据，但是训练过程还是不可控。比如：DataGenerator可以产生siamese数据，但是无法按照siamese的思想来训练。
"""

import os
import sys
project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_path)

from network import lenet3 as net
from datasets import dataset
import tensorflow as tf
from tensorflow.keras import layers
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


if __name__ == '__main__':
    images_path, images_label = util.get_dataset(g_datapath)
    num_class = len(set(images_label))
    batch_size = 110

    # 使用tf.data.Dataset.from_tensor_slices定义的数据集也可以用fit_generator训练。
    # dataset = sequence_dataset(images_path, images_label, num_class, batch_size=batch_size)  # ok
    dataset = dataset.DataGenerator(images_path, images_label, shape=(28, 28, 1), batch_size=32)

    model = tf.keras.Sequential([
        net.LeNet(),
        layers.Dense(num_class, activation='softmax')
    ])

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(dataset, epochs=5, steps_per_epoch=len(images_path) // batch_size + 1)

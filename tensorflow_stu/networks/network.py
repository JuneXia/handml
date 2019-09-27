# -*- coding: UTF-8 -*-
# /usr/
import os
import sys
project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_path)

import tensorflow as tf
import numpy as np

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

# Change Mark: 相对于lenet2.py，本次实验必须使用eager模式，否则会报错
tf.enable_eager_execution()
print('is eager executing: ', tf.executing_eagerly())

import numpy.random as rng

def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    # return K.variable(values,name=name)
    return tf.Variable(values,name=name)
#//TODO: figure out how to initialize layer biases in keras.
def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    # return K.variable(values,name=name)
    return tf.Variable(values,name=name)


class SiameseNet(tf.keras.Model):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    '''
    def build(self, input_shape):
        print(input_shape)
    '''

    def call(self, inputs, **kwargs):
        emb1 = self.embedding_net(inputs[0])
        emb2 = self.embedding_net(inputs[1])

        return emb1, emb2

    def get_output(self):
        print('[networks/network.py]:: SiameseNet.get_output]')
        return None


class SiameseBinClassifyNet(tf.keras.Model):
    """
    There are not save_weights function if inherit from tf.keras.layers.Layer
    """
    def __init__(self, embedding_net, imshape=(28, 28, 1)):
        super(SiameseBinClassifyNet, self).__init__()
        self.embedding_net = embedding_net
        self.l1_layer = tf.keras.layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
        self.fc1 = tf.keras.layers.Dense(1, activation='sigmoid')  # TODO: This fc1_layer should not in the siamese-network.
        self.predicts = None

        # TODO: Just to get model variable explicit
        data = np.ones((10, ) + imshape, dtype=np.float32)
        self.call((data, data))

    def build_failed(self, input_shape):
        '''
        for batch, (images, labels) in enumerate(self.validation_dataset):
            outputs = self.model(images)
            labels = labels.reshape((-1, 1))
            loss_step = self.loss_func(labels, outputs, (self.model.emb1, self.model.emb2))
        '''

        shape = tf.TensorShape((input_shape[1], self.output_dim))
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=True)
        # Be sure to call this at the end
        super(SiameseBinClassifyNet, self).build(input_shape)


    '''
    def build(self, input_shape):
        # The build method gets called the first time your layer is used.
        # Creating variables on build() allows you to make their shape depend
        # on the input shape and hence removes the need for the user to specify
        # full shapes. It is possible to create variables during __init__() if
        # you already know their full shapes.
        self.kernel = self.add_variable(
            "kernel", [input_shape[-1], self.output_units])
    '''

    def call(self, inputs, **kwargs):
        emb1 = self.embedding_net(inputs[0])
        emb2 = self.embedding_net(inputs[1])

        l1_dist = self.l1_layer([emb1, emb2])

        # l1_dist = tf.reduce_sum(tf.abs(emb1 - emb2), axis=1)
        # l1_dist = tf.expand_dims(l1_dist, axis=1)

        predicts = self.fc1(l1_dist)

        return predicts, (emb1, emb2)

    def get_output(self):
        return self.predicts


class ContrastiveBinClassifyNet(tf.keras.Model):
    def __init__(self):
        super(ContrastiveBinClassifyNet, self).__init__()
        self.L1_layer = tf.keras.layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
        self.fc1 = tf.keras.layers.Dense(1, activation='sigmoid')

    '''
    def build(self, input_shape):
        print(input_shape)
    '''

    def call(self, inputs, training=None, mask=None):
        emb1, emb2 = inputs

        L1_dist = self.L1_layer([emb1, emb2])

        # l1_dist = tf.reduce_sum(tf.abs(emb1 - emb2), axis=1)
        # l1_dist = tf.expand_dims(l1_dist, axis=1)

        predicts = self.fc1(L1_dist)

        return predicts, (emb1, emb2)

    def get_output(self):
        print('[networks/network.py]:: ContrastiveBinClassifyNet.get_output]')
        return None


class ClassificationNet(tf.keras.layers.Layer):
    def __init__(self, embedding_net, num_class):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        # self.nonlinear = tf.keras.activations.relu()  # not for custom train
        # self.nonlinear = tf.keras.layers.PReLU()
        # self.nonlinear = tf.keras.layers.Activation()
        self.nonlinear = tf.keras.layers.ReLU()
        self.fc1 = tf.keras.layers.Dense(num_class, activation='softmax')

    def call(self, inputs, **kwargs):
        output = self.embedding_net(inputs)
        output = self.nonlinear(output)
        scores = self.fc1(output)

        return scores


def create_model(just_embedding=False):
    # create embedding_net
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, [3, 3], activation='relu', input_shape=(None, None, 1)),
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.GlobalMaxPool2D(),
        tf.keras.layers.Dense(32)
    ])

    if not just_embedding:
        # 思路1：ok
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        model = tf.keras.Sequential([
            SiameseNet(model),
            ContrastiveBinClassifyNet()
        ])
        data = np.ones((10,) + (28, 28, 1), dtype=np.float32)
        model((data, data))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # 思路2：ok
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # model = SiameseBinClassifyNet(model)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    return model


if __name__ == '__main__1':
    model = create_model()

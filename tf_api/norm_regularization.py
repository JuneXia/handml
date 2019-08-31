#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


if __name__ == '__main__1':
    A = tf.constant([[1, 1], [2, 2], [3, 3]], dtype=tf.float32)

    with tf.Session() as sess:
        norm = tf.nn.l2_normalize(A, [0])  # 对列向量进行l2-norm
        # norm = tf.nn.l2_normalize(A, [1])  # 对行向量进行l2-norm

        arr = sess.run(norm)
        print(arr)
        sess.close()

if __name__ == '__main__2':
    A = tf.constant([[1, 1], [2, 2], [3, 3]], dtype=tf.float32)
    # A = tf.constant([1, 2, 3, 4], dtype=tf.float32)

    with tf.Session() as sess:
        # axis=0时，按行向量计算
        # ord控制的是p范数
        norm = tf.norm(A, ord=1, axis=0).eval()
        norm = tf.norm(A, ord=2, axis=0).eval()
        norm = tf.norm(A, ord='euclidean', axis=0).eval()
        norm = tf.norm(A, ord=np.inf, axis=0).eval()

        # axis=1时，按列向量计算
        # ord控制的是p范数
        norm = tf.norm(A, ord=1, axis=1).eval()
        norm = tf.norm(A, ord=2, axis=1).eval()
        norm = tf.norm(A, ord='euclidean', axis=1).eval()
        norm = tf.norm(A, ord=np.inf, axis=1).eval()

        # axis=-1时，表示按最后一个维度计算，这里即是按列向量计算
        # ord控制的是p范数
        norm = tf.norm(A, ord=1, axis=-1).eval()
        norm = tf.norm(A, ord=2, axis=-1).eval()
        norm = tf.norm(A, ord='euclidean', axis=-1).eval()
        norm = tf.norm(A, ord=np.inf, axis=-1).eval()

        # ord控制的是p范数，当axis=None时，tf.norm表示在得到p范数后的向量的基础上再做一次求和，然后再开根号。
        # 而此时当ord=2或者'euclidean'时，tf.norm得到的数值也称作Frobenius范数
        norm = tf.norm(A, ord=1, axis=None).eval()
        norm = tf.norm(A, ord=2, axis=None).eval()
        norm = tf.norm(A, ord='euclidean', axis=None).eval()
        norm = tf.norm(A, ord=np.inf, axis=None).eval()

        print(norm)


if __name__ == '__main__3':
    A = tf.constant([[1, 1], [2, 2], [3, 3]], dtype=tf.float32)
    B = tf.constant([1, 2, 3], dtype=tf.float32)

    with tf.Session() as sess:
        a = tf.nn.l2_loss(A).eval()
        b = tf.nn.l2_loss(B).eval()

        print(a)


if __name__ == '__main__4':
    sess = tf.Session()
    weight_decay = 0.1
    tmp = tf.constant([0, 1, 2, 3], dtype=tf.float32)
    """
    l2_reg=tf.contrib.layers.l2_regularizer(weight_decay)
    a=tf.get_variable("I_am_a",regularizer=l2_reg,initializer=tmp)
    """
    # **上面代码的等价代码
    a = tf.get_variable("I_am_a", initializer=tmp)
    a2 = tf.reduce_sum(a * a) * weight_decay / 2
    a3 = tf.get_variable(a.name.split(":")[0] + "/Regularizer/l2_regularizer", initializer=a2)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, a2)
    # **
    sess.run(tf.global_variables_initializer())
    keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    for key in keys:
        print("%s : %s" % (key.name, sess.run(key)))


if __name__ == '__main__5':
    sess = tf.Session()
    weight_decay = 0.1  # (1)定义weight_decay
    l2_reg = tf.contrib.layers.l2_regularizer(weight_decay)  # (2)定义l2_regularizer()
    tmp = tf.constant([0, 1, 2, 3], dtype=tf.float32)
    a = tf.get_variable("I_am_a", regularizer=l2_reg, initializer=tmp)  # (3)创建variable，l2_regularizer复制给regularizer参数。
    # 目测REXXX_LOSSES集合
    # regularizer定义会将a加入REGULARIZATION_LOSSES集合
    print("Global Set:")
    keys = tf.get_collection("variables")
    for key in keys:
        print(key.name)
    print("Regular Set:")
    keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    for key in keys:
        print(key.name)
    print("--------------------")
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    reg_set = tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES)  # (4)则REGULARIAZTION_LOSSES集合会包含所有被weight_decay后的参数和，将其相加
    l2_loss = tf.add_n(reg_set)
    print("loss=%s" % (sess.run(l2_loss)))
    """
    此处输出0.7,即:
       weight_decay*sigmal(w*2)/2=0.1*(0*0+1*1+2*2+3*3)/2=0.7
    其实代码自己写也很方便，用API看着比较正规。
    在网络模型中，直接将l2_loss加入loss就好了。(loss变大，执行train自然会decay)
    """


if __name__ == '__main__':
    def get_weights(shape, lambd):

        var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(var))
        return var


    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))
    batch_size = 8
    layer_dimension = [2, 10, 10, 10, 1]
    n_layers = len(layer_dimension)
    cur_lay = x
    in_dimension = layer_dimension[0]

    for i in range(1, n_layers):
        out_dimension = layer_dimension[i]
        weights = get_weights([in_dimension, out_dimension], 0.001)
        bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
        cur_lay = tf.nn.relu(tf.matmul(cur_lay, weights) + bias)
        in_dimension = layer_dimension[i]

    mess_loss = tf.reduce_mean(tf.square(y_ - cur_lay))
    tf.add_to_collection('losses', mess_loss)
    loss = tf.add_n(tf.get_collection('losses'))
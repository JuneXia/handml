#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


# tf.reduce_mean及其keepdims参数解析
if __name__ == '__main__1':
    # arr = tf.constant([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    arr = tf.constant([[1., 2.], [3., 4.]])
    # tile = tf.tile(arr, [3, 1])
    # retile = tf.transpose(tile)
    reduce_max = tf.reduce_mean(arr, 1, keepdims=True)
    sub = arr - reduce_max

    max = tf.reduce_max(arr, [0, 1])

    with tf.Session() as sess:
        print('debug')


# tf.tile张量扩展函数介绍
if __name__ == '__main__2':
    arr = tf.constant([[1., 2.], [3., 4.]])
    # tile = tf.tile(arr, [2])  # failed
    tile = tf.tile(arr, [2, 3])  # multiples列表中的元素个数必须要和arr的维度相等

    with tf.Session() as sess:
        print(sess.run(arr))
        print(sess.run(tile))


# tf.split, 可用于将tensor按块划分
if __name__ == '__main__3':
    # arr = tf.constant([[1., 2., 3.], [4., 5., 6.]])
    # arr = tf.ones([4, 4])

    onearr = np.ones([4, 4])
    for i, arr in enumerate(onearr):
        for j, a in enumerate(arr):
            onearr[i, j] = i * len(arr) + j

    arr = tf.constant(onearr)
    x = tf.split(arr, 2, 0)
    x_list = []
    for y in x:
        x_list.append(tf.split(y, 2, 1))

    with tf.Session() as sess:
        print(sess.run(arr))
        x_out = sess.run(x)
        print(x_out)


if __name__ == '__main__4':
    arr1 = tf.constant([[1., 2., 3.], [4., 5., 6.]])
    arr2 = tf.constant([[11., 21., 31.], [41., 51., 61.]])

    arr = tf.concat([arr1, arr2], 1)

    with tf.Session() as sess:
        arr = sess.run(arr)

        print('debug')


if __name__ == '__main__5':
    arr1 = tf.constant([[1., 2., 3.], [4., 5., 6.]])
    arr2 = tf.constant([[11., 21., 31.], [41., 51., 61.]])

    arr = tf.concat([arr1, arr2], 1)

    with tf.Session() as sess:
        arr = sess.run(arr)
        print('debug')


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import bigfloat


def numpy_softmax(x):
    shift_x = x - np.max(x)
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x)


def softmax_bf(x):
    exp_x = []
    for xi in x:
        arr = []
        for elem in xi:
            arr.append(bigfloat.exp(elem))
        exp_x.append(arr)
    sum_x = np.sum(exp_x)

    rslt = []
    for x in exp_x:
        arr = []
        for elem in x:
            arr.append(elem/sum_x)
        rslt.append(arr)

    soft_buf = []
    for rlt in rslt:
        arr = []
        for b in rlt:
            arr.append(b.__float__())
        soft_buf.append(arr)

    return soft_buf


def tfnnsoftmax_numstability(x):
    reduce_max = tf.reduce_max(x, 1, keepdims=True)
    prob = tf.nn.softmax(x - reduce_max)
    return prob


labels = np.array([[0.2, 0.3, 0.5],
                   [0.1, 0.6, 0.3]])  # 实际应用时，这个labels应该是one-hot编码
fc_output = np.array([[4, 1, -2],
                   [0.1, 1, 3]])


np_prob = numpy_softmax(fc_output)  # 使用numpy实现的softmax输出的概率
bf_prob = softmax_bf(fc_output)  # 使用bigfloat实现的softmax输出的概率
tfnn_prob = tf.nn.softmax(fc_output)  # 使用tf.nn.softmax输出的概率
tfnn_prob_stab = tfnnsoftmax_numstability(fc_output)  # 使用自己改进的数值稳定的tf.nn.softmax输出的概率


def pythonAPI_cross_entropy_loss(labels, softmax_array):
    cross_entropy = -labels * np.log(softmax_array)
    cross_ent_sum = np.sum(cross_entropy, 1)
    loss = np.mean(cross_ent_sum)
    return loss


def tf_baseAPI_cross_entropy_loss(labels, prob):
    '''
    随着训练的进行，模型准确率越来越高，softmax的输出概率在相异类别上越来越趋近于0，对这些趋近于0的数取log将会得到很大的数，即nan了。
    所以我们这里使用tf.clip_by_value对prob数值进行裁剪，过滤掉太小的prob值
    '''
    clip_prob = tf.clip_by_value(prob, 1e-10, 1.0)
    cross_entropy = -tf.reduce_sum(labels * tf.log(clip_prob), 1)
    loss = tf.reduce_mean(cross_entropy)

    return loss


# 推荐直接使用tf.nn.softmax_cross_entropy_with_logits接口，这是TensorFlow官方高度优化过的交叉熵。
def tf_nn_API_cross_entropy_loss(labels, fc_output):
    H = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=fc_output)
    loss = tf.reduce_mean(H)
    return loss


if __name__ == '__main__':
    with tf.Session() as sess:
        tfnn_prob_val = sess.run(tfnn_prob)
        tfnn_prob_stab_val = sess.run(tfnn_prob_stab)
        print('numpy实现的softmax输出概率: ', np_prob)
        print('bigfloat实现的softmax输出概率: ', bf_prob)
        print('tf.nn.softmax输出的概率: ', tfnn_prob_val)
        print('改进的tf.nn.softmax输出的概率: ', tfnn_prob_stab_val)

        npSoftmax_pyCrossEnt_loss = pythonAPI_cross_entropy_loss(labels, np_prob)
        bfSoftmax_pyCrossEnt_loss = pythonAPI_cross_entropy_loss(labels, bf_prob)
        tfnnSoftmax_pyCrossEnt_loss = pythonAPI_cross_entropy_loss(labels, tfnn_prob_val)
        tfnnSoftmaxStab_pyCrossEnt_loss = pythonAPI_cross_entropy_loss(labels, tfnn_prob_stab_val)
        print('npSoftmax_pyCrossEnt_loss：', npSoftmax_pyCrossEnt_loss)
        print('bfSoftmax_pyCrossEnt_loss：', bfSoftmax_pyCrossEnt_loss)
        print('tfnnSoftmax_pyCrossEnt_loss：', tfnnSoftmax_pyCrossEnt_loss)
        print('tfnnSoftmaxStab_pyCrossEnt_loss：', tfnnSoftmaxStab_pyCrossEnt_loss)

        npSoftmax_tfbaseAPI_CrossEnt_loss = tf_baseAPI_cross_entropy_loss(labels, np_prob)
        bfSoftmax_tfbaseAPI_CrossEnt_loss = tf_baseAPI_cross_entropy_loss(labels, bf_prob)
        tfnnSoftmax_tfbaseAPI_CrossEnt_loss = tf_baseAPI_cross_entropy_loss(labels, tfnn_prob_val)
        tfnnSoftmaxStab_tfbaseAPI_CrossEnt_loss = tf_baseAPI_cross_entropy_loss(labels, tfnn_prob_stab_val)
        print('npSoftmax_tfbaseAPI_CrossEnt_loss：', sess.run(npSoftmax_tfbaseAPI_CrossEnt_loss))
        print('bfSoftmax_tfbaseAPI_CrossEnt_loss：', sess.run(bfSoftmax_tfbaseAPI_CrossEnt_loss))
        print('tfnnSoftmax_tfbaseAPI_CrossEnt_loss：', sess.run(tfnnSoftmax_tfbaseAPI_CrossEnt_loss))
        print('tfnnSoftmaxStab_tfbaseAPI_CrossEnt_loss：', sess.run(tfnnSoftmaxStab_tfbaseAPI_CrossEnt_loss))

        tfnnAPI_loss = tf_nn_API_cross_entropy_loss(labels, fc_output)
        print('tfnnAPI_logits 交叉熵 loss：', sess.run(tfnnAPI_loss))

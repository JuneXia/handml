# -*- coding: UTF-8 -*-
import os
import sys
project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
project2_path = os.path.dirname(project_path)
print('[training]:: project_path: {}'.format(project_path))
print('[training]:: project2_path: {}'.format(project2_path))
sys.path.append(project_path)
sys.path.append(project2_path)

import tensorflow as tf
from tensorflow.python.keras import losses

from utils import dataset as datset
from utils import tools
import numpy as np
import math
from six import iteritems

from losses import loss

import socket
import getpass

home_path = os.environ['HOME']

user_name = getpass.getuser()
host_name = socket.gethostname()

if user_name in ['xiajun', 'yp']:
    g_datapath = os.path.join(home_path, 'res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44')
elif user_name in ['xiaj'] and host_name in ['ailab-server']:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # g_datapath = os.path.join(home_path, 'res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44')
    # g_datapath = '/disk2/res/VGGFace2/Experiment/mtcnn_align182x182_margin44'
    g_datapath = os.path.join(home_path, 'res/mnist/train')
elif user_name in ['xiaj'] and host_name in ['ubuntu-pc']:
    g_datapath = os.path.join(home_path, 'res/mnist')
else:
    print('unkown user_name:{}'.format(user_name))
    exit(0)

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)  # local 0.333
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)
print('is eager executing: ', tf.compat.v1.executing_eagerly())


class PrelogitsNormLoss(losses.Loss):
    def __init__(self, reduction="sum_over_batch_size", name=None):
        super(PrelogitsNormLoss, self).__init__(reduction=reduction, name=name)

        self.eps = 1e-4
        self.prelogits_norm_p = 1
        self.prelogits_norm_loss_factor = 5e-4
        # prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(model.output) + eps, ord=prelogits_norm_p, axis=1))
        # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * prelogits_norm_loss_factor)

    def call(self, y_true, y_pred):
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(y_pred) + self.eps, ord=self.prelogits_norm_p, axis=1))
        return prelogits_norm * self.prelogits_norm_loss_factor


def focal_loss(weights=None, alpha=0.25, gamma=2, name='focal_loss'):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """

    def _focal_loss1(y_true, y_pred):
        # sigmoid_p = tf.nn.sigmoid(prediction_tensor)
        # zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
        #
        # # For poitive prediction, only need consider front part loss, back part is 0;
        # # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
        # # target_tensor = tf.cast(target_tensor, dtype=tf.float32)
        # target_tensor = tf.one_hot(target_tensor, depth=prediction_tensor.get_shape().as_list()[-1])
        # pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
        #
        # # For negative prediction, only need consider back part loss, front part is 0;
        # # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        # neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
        # per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
        #                       - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))

        # y_true = tf.one_hot(y_true, depth=prediction_tensor.get_shape().as_list()[-1])
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        # 从logits计算softmax
        reduce_max = tf.reduce_max(y_pred, axis=1, keepdims=True)
        y_pred = tf.nn.softmax(y_pred - reduce_max)

        # 计算交叉熵
        clip_preb = tf.clip_by_value(y_pred, 1e-10, 1.0)
        cross_entropy = -tf.reduce_sum(y_true * tf.log(clip_preb), 1)

        # 计算focal_loss
        prob = tf.reduce_max(y_pred, axis=1)
        weight = tf.pow(tf.subtract(1., prob), gamma)
        fl = tf.multiply(tf.multiply(weight, cross_entropy), alpha)
        loss = tf.reduce_sum(fl, name=name)

        return loss


    def _focal_loss2(y_true, y_pred):
        # sigmoid_p = tf.nn.sigmoid(prediction_tensor)
        # zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
        #
        # # For poitive prediction, only need consider front part loss, back part is 0;
        # # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
        # # target_tensor = tf.cast(target_tensor, dtype=tf.float32)
        # target_tensor = tf.one_hot(target_tensor, depth=prediction_tensor.get_shape().as_list()[-1])
        # pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
        #
        # # For negative prediction, only need consider back part loss, front part is 0;
        # # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        # neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
        # per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
        #                       - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))

        # y_true = tf.one_hot(y_true, depth=prediction_tensor.get_shape().as_list()[-1])
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


        # 从logits计算softmax
        reduce_max = tf.reduce_max(y_pred, axis=1, keepdims=True)
        y_pred = tf.nn.softmax(y_pred - reduce_max)

        # 计算交叉熵
        # clip_preb = tf.clip_by_value(y_pred, 1e-10, 1.0)
        # cross_entropy = -tf.reduce_sum(y_true * tf.log(clip_preb), 1)

        # 计算focal_loss
        prob = tf.reduce_max(y_pred, axis=1)
        weight = tf.pow(tf.subtract(1., prob), gamma)
        fl = tf.multiply(tf.multiply(weight, cross_entropy), alpha)
        loss = tf.reduce_sum(fl, name=name)

        return loss

    def _focal_loss(y_true, y_pred):
        # 从logits计算softmax
        reduce_max = tf.reduce_max(y_pred, axis=1, keepdims=True)
        y_pred = tf.nn.softmax(tf.subtract(y_pred, reduce_max))

        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        y_pred = tf.clip_by_value(y_pred, 1e-10, 1.0)
        # cross_entropy = -tf.reduce_sum(y_true * tf.log(y_pred), 1)
        cross_entropy = -tf.reduce_sum(tf.multiply(y_true, tf.log(y_pred)), axis=1)

        # 计算focal_loss
        prob = tf.reduce_max(y_pred, axis=1)
        weight = tf.pow(tf.subtract(1., prob), gamma)
        # weight = tf.multiply(tf.multiply(weight, y_true), alpha)
        # weight = tf.reduce_max(weight, axis=1)

        fl = tf.multiply(tf.multiply(weight, cross_entropy), alpha)
        loss = tf.reduce_sum(fl, name=name)

        return loss

    return _focal_loss


def multi_category_focal_loss2(gamma=2., alpha=.25):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha控制真值y_true为1/0时的权重
        1的权重为alpha, 0的权重为1-alpha
    当你的模型欠拟合，学习存在困难时，可以尝试适用本函数作为loss
    当模型过于激进(无论何时总是倾向于预测出1),尝试将alpha调小
    当模型过于惰性(无论何时总是倾向于预测出0,或是某一个固定的常数,说明没有学到有效特征)
        尝试将alpha调大,鼓励模型进行预测出1。
    Usage:
     model.compile(loss=[multi_category_focal_loss2(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    def multi_category_focal_loss2_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
        ce = -tf.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
        loss = tf.reduce_mean(fl)
        return loss

    return multi_category_focal_loss2_fixed


def cosineface_losses(embedding, labels, out_num, w_init=None, s=30., m=0.4):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value, default is 30
    :param out_num: output class num
    :param m: the margin value, default is 0.4
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    with tf.variable_scope('cosineface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos_theta - m
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t_m = tf.subtract(cos_t, m, name='cos_t_m')

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        output = tf.add(s * tf.multiply(cos_t, inv_mask), s * tf.multiply(cos_t_m, mask), name='cosineface_loss_output')
    return output


def center_loss2(normalized_pred, logits_w, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    # nrof_features = normalized_pred.get_shape()[1]
    # centers = tf.get_variable('centers', [nrof_classes, 5], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)

    logits_w = tf.nn.l2_normalize(logits_w, 0, 1e-10)
    centers = tf.transpose(logits_w)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - normalized_pred)
    centers = tf.scatter_sub(centers, label, diff)
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(normalized_pred - centers_batch))
    return loss, centers

""
def arcface_loss1(embedding, labels, out_num, w_init=None, s=64., m=0.5, name='arc_loss'):
    '''
    Reference: https://github.com/auroua/InsightFace_TF/losses/face_losses.py
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope('Logits', reuse=True):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s*(cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        # mask = labels
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')

    ross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output)
    loss = tf.reduce_mean(ross_entropy, name=name)
    return loss
""

def arcface_loss(embedding, labels, out_num, w_init=None, s=64., m=0.5, name='arc_loss'):
    '''
    Reference: https://github.com/auroua/InsightFace_TF/losses/face_losses.py
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    shape_list = labels.shape.as_list()
    if len(shape_list) == 2 and shape_list[-1] > 1:
        onehot = True
    else:
        onehot = False

    with tf.variable_scope('Logits', reuse=True):
        weights = tf.get_variable(name='weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)

    # embedding = tf.nn.l2_normalize(embedding, 0, 1e-10)
    weights = tf.nn.l2_normalize(weights, 0, 1e-10)

    # cos(theta+m)
    fc7 = tf.matmul(embedding, weights, name='cos_t')  # cos_t

    if onehot:
        labs = tf.reshape(tf.argmax(labels, axis=1), (-1, 1))
    else:
        labs = tf.reshape(labels, (-1, 1))
    zy = tf.gather_nd(fc7, labs, batch_dims=1)  # cos_t_flatten

    # theta = tf.acos(0.707106781)*180/np.pi  == 45
    theta = tf.acos(zy)
    zy_margin = tf.cos(theta + m)

    if not onehot:
        onehot_labels = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
    else:
        onehot_labels = labels
    diff = zy_margin - zy
    diff = tf.expand_dims(diff, 1)
    fc7 = fc7 + tf.multiply(onehot_labels, diff)
    output = fc7 * s

    ross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output)
    loss = tf.reduce_mean(ross_entropy, name=name)
    return loss


def arcface_loss_仿照csdnMX版本编写(embedding, labels, out_num, w_init=None, s=64., m=0.5, name='arc_loss'):
    '''
    Reference: https://github.com/auroua/InsightFace_TF/losses/face_losses.py
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope('Logits', reuse=True):
        # inputs and weights norm

        # embedding normalize
        # embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        # embedding = tf.div(embedding, embedding_norm, name='norm_embedding')

        embedding = tf.nn.l2_normalize(embedding, 0, 1e-10) * s

        # 定义weight, 并normalize
        weights = tf.get_variable(name='weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        # weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        # weights = tf.div(weights, weights_norm, name='norm_weights')

        weights = tf.nn.l2_normalize(weights, 0, 1e-10)

        # cos(theta+m)
        fc7 = tf.matmul(embedding, weights, name='cos_t')  # cos_t

        ONE_HOT = False
        if ONE_HOT:
            labs = tf.reshape(tf.argmax(labels, axis=1), (-1, 1))
        else:
            labs = tf.reshape(labels, (-1, 1))
        zy = tf.gather_nd(fc7, labs, batch_dims=1)  # cos_t_flatten

        cos_t = zy/s

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        new_zy = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        zy_keep = zy - s*mm
        new_zy = tf.where(cond, new_zy, zy_keep)

        diff = new_zy - zy
        diff = tf.expand_dims(diff, 1)
        onehot_labels = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        body = tf.multiply(onehot_labels, diff)
        output = fc7 + body



        # keep_val = s*(cos_t_flatten - mm)
        # cos_mt_temp = tf.where(cond, cos_mt, keep_val)
        #
        # diff = cos_mt_temp - s * cos_t_flatten
        # diff = tf.expand_dims(diff, axis=1)
        # # onehot_labels = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        # diff = tf.reshape(diff, (-1, 1))
        # body = tf.multiply(labels, diff)
        #
        # output = body + cos_t * s

    ross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output)
    loss = tf.reduce_mean(ross_entropy, name=name)
    return loss


def arcface_loss_待调试(embedding, labels, out_num, w_init=None, s=64., m=0.5, name='arc_loss'):
    '''
    Reference: https://github.com/auroua/InsightFace_TF/losses/face_losses.py
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope('Logits', reuse=True):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')

        labs = tf.reshape(tf.argmax(labels, axis=1), (-1, 1))
        cos_t_flatten = tf.gather_nd(cos_t, labs, batch_dims=1)

        cos_t2 = tf.square(cos_t_flatten, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t_flatten, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t_flatten - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s*(cos_t_flatten - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        diff = cos_mt_temp - s * cos_t_flatten
        diff = tf.expand_dims(diff, axis=1)
        # onehot_labels = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        diff = tf.reshape(diff, (-1, 1))
        body = tf.multiply(labels, diff)

        output = body + cos_t * s

    ross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output)
    loss = tf.reduce_mean(ross_entropy, name=name)
    return loss



def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss


# embedding = np.ones((3, 5), dtype=np.float32)
# labels = np.array([1, 0, 2])
# arcface_loss(embedding, labels=labels, out_num=10)


def prelogits_norm_loss(y_pred, from_logits=False):
    eps = 1e-4
    prelogits_norm_p = 1
    prelogits_norm_loss_factor = 5e-4
    prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(y_pred) + eps, ord=prelogits_norm_p, axis=1))
    # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * prelogits_norm_loss_factor)
    return prelogits_norm * prelogits_norm_loss_factor


class CustomModel(tf.keras.Model):
    def __init__(self, n_classes, conv_base):
        super(CustomModel, self).__init__(name='my_model')
        self.total_loss = []

        self.conv_base = get_conv_base_model(conv_base=conv_base, imshape=(160, 160, 3))

        self.conv_base.trainable = False
        # conv_base.summary()

        bottleneck_size = 512
        weight_decay = 5e-4

        self.dropout = tf.keras.layers.Dropout(0.5, seed=666)

        kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
        self.dense_bottleneck = tf.keras.layers.Dense(bottleneck_size, kernel_regularizer=kernel_regularizer)
        self.prelogits = tf.keras.layers.BatchNormalization(name='Bottleneck')
        # self.prelogits = tf.keras.layers.Activation('relu')

        kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
        self.dense2 = tf.keras.layers.Dense(n_classes, kernel_regularizer=kernel_regularizer)

    def call(self, inputs, **kwargs):
        output = self.conv_base(inputs)
        output = self.dropout(output)
        output = self.dense_bottleneck(output)
        prelogits = self.prelogits(output)
        output = self.dense2(prelogits)

        return prelogits, output

    def compute_output_shape(self, input_shape):
        print(input_shape)


def sequential_model(conv_base, weight_decay, bottleneck_size, n_classes):
    model = tf.keras.Sequential()
    model.add(conv_base)
    model.add(tf.keras.layers.Dropout(0.5, seed=666))

    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    model.add(tf.keras.layers.Dense(bottleneck_size, kernel_regularizer=kernel_regularizer))
    model.add(tf.keras.layers.BatchNormalization(name='Bottleneck'))
    # model.add(tf.keras.layers.Activation('relu'))

    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    model.add(tf.keras.layers.Dense(n_classes, kernel_regularizer=kernel_regularizer))

    return model


def functional_model(conv_base, input_shape, weight_decay, bottleneck_size, n_classes):
    iminputs = tf.keras.Input(shape=input_shape, name='input')
    model = conv_base(iminputs)
    # model = tf.keras.layers.Dropout(0.5, seed=666)(model)

    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    model = tf.keras.layers.Dense(bottleneck_size, kernel_regularizer=kernel_regularizer)(model)
    prelogits = tf.keras.layers.BatchNormalization(name='Bottleneck')(model)
    prelogits = tf.keras.layers.Activation('relu')(prelogits)

    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    output = tf.keras.layers.Dense(n_classes, kernel_regularizer=kernel_regularizer)(prelogits)
    # output = tf.keras.layers.Activation('softmax')(output)

    if datset.MULTI_OUTPUT:
        model = tf.keras.Model(inputs=iminputs, outputs=[prelogits, output])
    else:
        model = tf.keras.Model(inputs=iminputs, outputs=[output])

    return model


def custom_model(n_classes, input_shape=(28, 28, 1)):
    conv_base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    # conv_base = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    # conv_base = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    conv_base.trainable = True
    # conv_base.summary()
    #for layer in conv_base.layers[:-4]:
    #    layer.trainable = False

    bottleneck_size = 512
    weight_decay = 5e-4

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # model = sequential_model(conv_base, weight_decay, bottleneck_size, n_classes)  # OK
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model = functional_model(conv_base, input_shape, weight_decay, bottleneck_size, n_classes)  # OK
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    return model


def scheduler(epoch):
    # if epoch < 3:
    #     return 0.1
    if epoch < 10:
        return 0.05
    elif epoch < 20:
        return 0.005
    elif epoch < 30:
        return 0.0005
    elif epoch < 40:
        return 0.0001

    if epoch < 100:
        return 0.05
    elif epoch < 200:
        return 0.005
    elif epoch < 276:
        return 0.0005
    else:
        # return 0.0005 * np.exp(0.01 * (70 - epoch))
        return 0.0001


def scheduler_experiment(epoch):
    if epoch < 5:
        return 0.05
    elif epoch < 10:
        return 0.005
    elif epoch < 15:
        return 0.0005
    else:
        # return 0.0005 * np.exp(0.01 * (70 - epoch))
        return 0.0001


class Reporter(object):
    def __init__(self):
        big_samples = 3000000
        big_epochs = 100
        tiny_samples = 1000
        tiny_epochs = 5
        k = (big_epochs-tiny_epochs)/(big_samples-tiny_samples)
        b = tiny_epochs - big_epochs*k
        y = k*100000+b
        print(y)

        self.records = {}
        self.current_key = ''

    def add_record(self, key, value):
        self.records[key] = value

    def add_records(self, params_dict):
        for key, value in iteritems(params_dict):
            self.add_record(key, value)

    def malloc_record(self, key):
        self.records[key] = []
        self.current_key = key

    def record_cb(self, line):
        self.records[self.current_key].append(line)

    def save_record(self, save_file):
        records = list(self.records.items())
        records.sort()
        with open(save_file, 'w') as f:
            for key, value in records:
                if type(value) == list:
                    f.write('\n%s: \n' % key)
                    for val in value:
                        f.write('%s\n' % str(val))
                    f.write('\n')
                else:
                    f.write('%s: %s\n' % (key, str(value)))


def toy_model(input_shape=(28, 28, 1)):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, [3, 3], activation='relu', input_shape=(None, None, input_shape[2])),
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.GlobalMaxPool2D(),
        tf.keras.layers.Dense(32)
    ])

    return model


def get_conv_base_model(conv_base, imshape):
    if conv_base == 'MobileNetV2':
        conv_base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False,
                                                            input_shape=imshape, pooling='avg')
    elif conv_base == 'Xception':
        conv_base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False,
                                                         input_shape=imshape, pooling='avg')
    elif conv_base == 'InceptionResnetV2':
        conv_base_model = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False,
                                                                  input_shape=imshape, pooling='avg')
    elif conv_base == 'DenseNet201':
        conv_base_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False,
                                                            input_shape=imshape, pooling='avg')
    elif conv_base == 'ResNet50':
        conv_base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False,
                                                         input_shape=imshape, pooling='avg')
    else:
        conv_base_model = toy_model(imshape)

    return conv_base_model


class Trainer(object):
    def __init__(self):
        self.params = {}  # TODO： 将所有的参数都放在着一个字典里可能不太合适，感觉还是多分几个字典比较好。
        self.params['executing_eagerly'] = tf.executing_eagerly()
        self.datas = {}

    def get_params(self):
        return self.params

    def get_logdir(self):
        return self.log_dir

    def load_dataset(self, data_path, min_nrof_cls=10, max_nrof_cls=4000, validation_ratio=0.1):
        train_images_path, train_images_label, validation_images_path, validation_images_label = datset.load_dataset(
            data_path, min_nrof_cls=min_nrof_cls, max_nrof_cls=max_nrof_cls, validation_ratio=validation_ratio)
        num_train_images = len(train_images_path)
        num_validation_images = len(validation_images_path)
        num_classes = len(set(train_images_label))

        self.datas['train_images_path'] = train_images_path
        self.datas['train_images_label'] = train_images_label
        self.datas['validation_images_path'] = validation_images_path
        self.datas['validation_images_label'] = validation_images_label

        self.params['data_path'] = data_path
        self.params['min_nrof_cls'] = min_nrof_cls
        self.params['max_nrof_cls'] = max_nrof_cls
        self.params['validation_ratio'] = validation_ratio
        self.params['num_train_images'] = num_train_images
        self.params['num_validation_images'] = num_validation_images
        self.params['num_classes'] = num_classes

    def shuffle_data(self, purpose, shuffle=True):
        '''
        在自定义训练中，就算设置reshuffle_each_iteration=True，每个epoch迭代完成后数据也并不会reshuffle.
        :param purpose:
        :param shuffle:
        :return:
        '''
        if purpose == 'train':
            filenames = tf.constant(self.datas['train_images_path'])
            labels = tf.constant(self.datas['train_images_label'])
            parse_func = self.imparse.train_parse_func
            buffer_size = min(self.params['num_train_images'], 1000)
        elif purpose == 'val':
            filenames = tf.constant(self.datas['validation_images_path'])
            labels = tf.constant(self.datas['validation_images_label'])
            parse_func = self.imparse.validation_parse_func
            buffer_size = min(self.params['num_validation_images'], 1000)
        else:
            raise Exception('purpose just support train or val. get: {}'.format(purpose))

        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(parse_func, num_parallel_calls=4)  # tf.data.experimental.AUTOTUNE
        dataset = dataset.shuffle(buffer_size=buffer_size,
                                  # seed=tf.compat.v1.set_random_seed(666),
                                  reshuffle_each_iteration=True
                                  ).batch(self.params['batch_size'])  # repeat 不指定参数表示允许无穷迭代

        if purpose == 'train':
            self.train_dataset = dataset.prefetch(buffer_size=buffer_size)
        elif purpose == 'val':
            self.validation_dataset = dataset.prefetch(buffer_size=buffer_size)

    def set_train_params(self, imshape=(160, 160, 3), batch_size=96, max_epochs=1):
        self.params['max_epochs'] = max_epochs
        self.params['imshape'] = imshape
        self.params['batch_size'] = batch_size
        self.params['max_epochs'] = max_epochs
        if user_name in ['xiaj'] and host_name in ['ailab-server']:
            # TODO: ailab训练速度慢，而且facenet源码epoch_size也是等于1000，故这里暂且用1000吧。
            self.params['train_steps_per_epoch'] = 1000
        else:
            self.params['train_steps_per_epoch'] = tools.steps_per_epoch(self.params['num_train_images'], batch_size, allow_less_batsize=False)
        self.params['validation_steps_per_epoch'] = tools.steps_per_epoch(self.params['num_validation_images'], batch_size, allow_less_batsize=False)
        print('train_steps_per_epoch={}'.format(self.params['train_steps_per_epoch']))
        print('validation_steps_per_epoch={}'.format(self.params['validation_steps_per_epoch']))

        self.imparse = datset.ImageParse(imshape=imshape, n_classes=self.params['num_classes'])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.shuffle_data('train')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.shuffle_data('val')
        # filenames = tf.constant(self.datas['validation_images_path'])
        # labels = tf.constant(self.datas['validation_images_label'])
        # validation_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        # validation_dataset = validation_dataset.map(self.imparse.validation_parse_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # self.validation_dataset = validation_dataset.shuffle(buffer_size=min(self.params['num_validation_images'], 1000),
        #                                                 seed=tf.compat.v1.set_random_seed(666),
        #                                                 reshuffle_each_iteration=True).batch(batch_size).repeat(1).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_model(self, conv_base='toy_model', custom_model=False):
        self.params['conv_base'] = conv_base
        self.params['bottleneck_size'] = 512
        self.params['weight_decay'] = 5e-4

        if custom_model:
            self.model = CustomModel(self.params['num_classes'], conv_base)
            imshape = self.params['imshape']
            self.model.build(input_shape=(None, imshape[0], imshape[1], imshape[2]))
        else:
            conv_base_model = get_conv_base_model(conv_base, self.params['imshape'])
            conv_base_model.trainable = False

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # model = sequential_model(conv_base_model, self.params['imshape'], self.params['weight_decay'], self.params['bottleneck_size'], self.params['num_classes'])  # OK
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.model = functional_model(conv_base_model, self.params['imshape'], self.params['weight_decay'],
                                          self.params['bottleneck_size'], self.params['num_classes'])  # OK
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        print(self.model.summary())
        print('Model trainable variables:')
        print('{:30s} {:20s} {:20s}'.format('name', 'shape', 'dtype'))
        print('_________________________________________________________________')
        for variable in self.model.trainable_variables:
            print('{:30s} {:20s} {:20s}'.format(variable.name, str(variable.get_shape().as_list()), str(variable.dtype)))
        print('_________________________________________________________________')


    def set_record(self):
        self.save_basename = tools.strcat([tools.get_strtime(), self.params['conv_base'], 'n_cls:' + str(self.params['num_classes'])], cat_mark='-')
        self.log_dir = os.path.join('logs', self.save_basename)
        self.model_save_dir = os.path.join('save_model', self.save_basename)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        model_ckpt_dir = os.path.join(self.model_save_dir, 'ckpt')
        if not os.path.exists(model_ckpt_dir):
            os.makedirs(model_ckpt_dir)
        self.model_ckpt_dir = os.path.join(model_ckpt_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')

        self.callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
                          tf.keras.callbacks.TensorBoard(self.log_dir, histogram_freq=0),  # TODO: 对于非eager模式，histogram_freq不能设为1
                          # tf.keras.callbacks.ModelCheckpoint(self.model_ckpt_dir, verbose=1, save_best_only=True, save_freq=2)  # save_freq>1时 运行时会崩溃。
                          ]

    def set_loss(self):
        # self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
        # self.loss_func = tf.keras.losses.CategoricalCrossentropy()
        # self.loss_func = loss.ComplexLoss2()
        self.cross_loss_func = tf.keras.losses.CategoricalCrossentropy()
        # self.focal_loss_func = multi_category_focal_loss2(alpha=2.0, gamma=0.25)
        self.focal_loss_func = focal_loss()
        self.prelogits_loss_func = PrelogitsNormLoss()

    def set_metric(self):
        # self.metric_func = tf.keras.metrics.SparseCategoricalAccuracy()
        self.metric_func = tf.keras.metrics.CategoricalAccuracy()

    def set_optimizer(self):
        self.learning_rate = 0.005  # src = 0.001
        # learning_rate = tf.train.exponential_decay(0.01, global_step=self.global_step, decay_steps=2, decay_rate=0.03)
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    def compile(self, optimizer, loss, metrics):
        pass

    def set_loss_metric(self, losses=[], metrics=[]):
        raise Exception('使用set_loss、set_metric、set_optimizer代替')

        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=self.optimizer,
                           loss=losses,
                           metrics=metrics)

        self.params['losses'] = losses
        self.params['metrics'] = metrics

    def fit(self):
        history = self.model.fit(self.train_dataset,
                            steps_per_epoch=self.params['train_steps_per_epoch'],
                            epochs=self.params['max_epochs'],
                            validation_data=self.validation_dataset,
                            validation_steps=self.params['validation_steps_per_epoch'],
                            callbacks=self.callbacks,
                            validation_freq=5,
                            workers=4,
                            use_multiprocessing=True
                            )

    def custom_fit(self):
        """
        tf.data数据迭代参考：https://blog.csdn.net/qq_34914551/article/details/96834647
        :return:
        """
        epoches = 300
        debug_multi_loss = False
        for epoch in range(epoches):
            # lr = self.optimizer.learning_rate.assign(scheduler(epoch))
            # print('Epoch: {}/{}, learning-rate: {}'.format(epoch, epoches, lr.numpy()))

            # for metric in self.metrics:
            #     metric.reset_states()

            # step = 0
            # while True:
            #     try:
            #         images, labels = next(train_iter)
            #     except StopIteration:
            #         # self.shuffle_data('train')
            #         break
            #     if step < 2:
            #         print(labels.numpy().argmax(axis=1))

            # train_iter = iter(self.train_dataset)
            # train_iter = tfe.Iterator(self.train_dataset)
            self.metric_func.reset_states()
            for step, (images, labels) in enumerate(self.train_dataset):
                if step < 2:
                    print(labels.numpy().argmax(axis=1))

                with tf.GradientTape() as t:
                    outputs = self.model(images)

                    # debug
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    if False:
                        batch_size = 3
                        self.params['num_classes'] = 7
                        emb_size = 5
                        embedding = np.ones((batch_size, emb_size), dtype=np.float32)
                        embedding[:, 1::2] = 0
                        labels = np.arange(self.params['num_classes'])
                        np.random.shuffle(labels)
                        # labels = labels[0:batch_size]
                        labels = np.array([6, 0, 3], dtype=np.int32)
                    else:
                        embedding = outputs[0]
                        labels = tf.argmax(labels, axis=1)



                    # arcloss1 = arcface_loss1(embedding, labels, self.params['num_classes'], w_init=tf.ones_initializer)
                    arcloss = arcface_loss(embedding, labels, self.params['num_classes'], w_init=tf.ones_initializer)
                    # crossloss = self.focal_loss_func(labels, outputs[1])
                    triplet_loss, fraction_positive_triplets = batch_all_triplet_loss(labels, embedding, 0.5, squared=False)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



                    # if type(outputs) != np.ndarray:
                    #     outputs = outputs.numpy()
                    # labels = labels.reshape((-1, 1))
                    step_crossloss = self.cross_loss_func(labels, outputs)
                    # step_crossloss = self.focal_loss_func(labels, outputs)
                    step_loss = step_crossloss
                    if debug_multi_loss:
                        step_logitloss = self.prelogits_loss_func(labels, outputs)
                        step_loss = step_loss + step_logitloss

                if debug_multi_loss:
                    grads = t.gradient([step_loss, step_loss],
                                       [self.model.trainable_variables[:-2], self.model.trainable_variables[-2:]])
                    self.optimizer.apply_gradients(zip(grads[0], self.model.trainable_variables[:-2]))
                    self.optimizer.apply_gradients(zip(grads[1], self.model.trainable_variables[-2:]))
                else:
                    grads = t.gradient(step_loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                rslt = self.metric_func(labels, outputs)
                # self.metric_func.result()
                if True or step % 50 == 0:
                    print('Training: step: {}, epoch: {}, loss: {}, logitsloss: {}, acc: {}'.format(step,
                                                                                                    epoch,
                                                                                                    step_loss.numpy(),
                                                                                                    0,
                                                                                                    rslt.numpy()))

                # step += 1

                # for metric in self.metrics:
                #     rslt = metric(labels, outputs, loss_step)
                #     tf.contrib.summary.scalar('train_' + metric.name(), rslt)
                #
                # if step % 10 == 0:
                #     print('Training: global step: {}, epoch: {}/{}, num_batch: {}'.format(self.global_step.numpy(), epoch,
                #                                                                           self.epoch_size,
                #                                                                           step), end=' ')
                #     for metric in self.metrics:
                #         print(metric.name(), metric.result().numpy(), end=' ')
                #     print('')

            self.metric_func.reset_states()
            for step, (images, labels) in enumerate(self.validation_dataset):
                outputs = self.model(images)
                # if type(outputs) != np.ndarray:
                #     outputs = outputs.numpy()
                # labels = labels.reshape((-1, 1))
                step_crossloss = self.cross_loss_func(labels, outputs)
                # step_focalloss = self.focal_loss_func(labels, outputs)
                step_loss = step_crossloss  # + 0.5*step_focalloss
                if debug_multi_loss:
                    step_logitloss = self.prelogits_loss_func(labels, outputs)
                    step_loss = step_loss + step_logitloss

                rslt = self.metric_func(labels, outputs)
                # self.metric_func.result()
                if True or step % 50 == 0:
                    print('Validation: step: {}, epoch: {}, loss: {}, logitsloss: {}, acc: {}'.format(step,
                                                                                                    epoch,
                                                                                                    step_loss.numpy(),
                                                                                                    0,
                                                                                                    rslt.numpy()))



            # if epoch == 20:
            #     self.finetune(-5)
            # elif epoch == 50:
            #     self.finetune(-8)
            # elif epoch == 60:
            #     self.finetune(-15)
            # elif epoch == 100:
            #     self.finetune(-30)
            # elif epoch == 150:
            #     self.finetune(-60)
            # elif epoch == 200:
            #     self.finetune(-100)
    def save_model(self):
        # TODO: 临时写法
        try:
            self.model.save(os.path.join(self.model_save_dir, 'Model.h5'))
            checkpoint_path = os.path.join(self.model_save_dir, 'ckpt')
            self.model.save_weights(checkpoint_path)
        except:
            checkpoint_path = os.path.join(self.model_save_dir, 'ckpt')
            self.model.save_weights(checkpoint_path)

    def finetune(self, num_layers):
        conv_base_model = self.model.layers[1]
        conv_base_model.trainable = True
        for layer in conv_base_model.layers[:num_layers]:
            layer.trainable = False

        print('finetune {} layers!'.format(num_layers))
        print('\n****************************************************')
        print('layer.name \t layer.trainable')
        for l in conv_base_model.layers:
            print(l.name, '\t', l.trainable)
        print('****************************************************\n')
        conv_base_model.summary()


if __name__ == '__main__1':
    reporter = Reporter()
    trainer = Trainer()
    trainer.load_dataset(g_datapath, min_nrof_cls=10, max_nrof_cls=4000, validation_ratio=0.02)
    # trainer.load_dataset(g_datapath, min_nrof_cls=10, max_nrof_cls=4000, validation_ratio=0.1)
    trainer.set_train_params(imshape=(28, 28, 3), batch_size=96, max_epochs=2)
    # trainer.set_train_params(imshape=(160, 160, 3), batch_size=96, max_epochs=276)
    trainer.set_model(custom_model=True)  # InceptionResnetV2, MobileNetV2, Xception, ResNet50
    trainer.set_record()
    if datset.MULTI_OUTPUT:
        trainer.set_loss_metric(losses=[prelogits_norm_loss, 'sparse_categorical_crossentropy'], metrics=[[], 'accuracy'])
    else:
        # trainer.set_loss_metric(losses=['sparse_categorical_crossentropy'], metrics=['accuracy'])
        trainer.set_loss()
        trainer.set_metric()
        trainer.set_optimizer()
    reporter.add_records(trainer.get_params())

    trainer.custom_fit()
    # trainer.finetune()
    trainer.save_model()
    reporter.save_record(os.path.join(trainer.get_logdir(), 'params.txt'))


# MovingAverage在Tensorflow2中可能需要在自定义训练中才能实现，或者自己继承tf的相关类来实现。
# tf.moving_average_variables

# 单输出模型似乎比多输出模型训练快


if __name__ == '__main__2':
    def func(x, y):
        return x*y

    f = lambda x, y, z: func(x, y)+z
    a = f(1, 2, 3)
    print(a)



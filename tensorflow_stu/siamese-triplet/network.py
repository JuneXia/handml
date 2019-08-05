from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def weights_variable(shape, name=None):
    # initializer = tf.glorot_uniform_initializer(seed=None, dtype=tf.float32)  # 梯度爆炸
    initializer = tf.glorot_normal_initializer(seed=tf.set_random_seed(1), dtype=tf.float32)  # 梯度爆炸
    # initializer = tf.variance_scaling_initializer(seed=tf.set_random_seed(1))
    # return tf.get_variable('weights', shape=shape)
    # return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weights')
    name = 'weights' if name is None else name
    return tf.get_variable(name, shape=shape, initializer=initializer)


def biases_variable(shape):
    # initializer = tf.glorot_uniform_initializer(seed=None, dtype=tf.float32)
    initializer = tf.glorot_normal_initializer(seed=tf.set_random_seed(1), dtype=tf.float32)
    # initializer = tf.variance_scaling_initializer(seed=tf.set_random_seed(1))
    # return tf.get_variable('biases', shape=shape)
    # return tf.Variable(tf.zeros(shape), name='biases')
    return tf.get_variable('biases', shape=shape, initializer=initializer)


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        '''
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels//groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])
        '''
        weights = weights_variable([filter_height, filter_width, input_channels // groups, num_filters])
        biases = biases_variable([num_filters])

        if groups == 1:
            conv = convolve(x, weights)

        # In the cases of multiple groups, split inputs & weights and
        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                     value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(axis=3, values=output_groups)

        # Add biases
        # bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
        bias = tf.nn.bias_add(conv, biases)
        # bias = tf.reshape(rslt, tf.shape(conv))

        # bias = batch_norm(bias, True)

        # Apply relu function
        relu = tf.nn.relu(bias, name='relu')

    return relu


def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases
        '''
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)
        '''
        weights = weights_variable([num_in, num_out])
        biases = biases_variable([num_out])

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu:
            # act = batch_norm(act, True, False)

            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def LeNet(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='LeNet'):
    """Creates the Inception Resnet V1 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
    end_points = {}

    with tf.variable_scope(scope, 'LeNet', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
                # 149 x 149 x 32
                net = slim.conv2d(inputs, 6, 5, stride=1, padding='SAME',
                                  scope='Conv2d_1a_3x3')
                end_points['Conv2d_1a_3x3'] = net

                # 73 x 73 x 64
                net = slim.max_pool2d(net, 2, stride=2, padding='SAME',
                                      scope='MaxPool_1a_2x2')
                end_points['MaxPool_3a_3x3'] = net

                # 147 x 147 x 32
                net = slim.conv2d(net, 16, 5, padding='SAME',
                                  scope='Conv2d_2a_3x3')
                end_points['Conv2d_2a_3x3'] = net

                # 73 x 73 x 64
                net = slim.max_pool2d(net, 2, stride=2, padding='SAME',
                                      scope='MaxPool_2a_3x3')
                end_points['MaxPool_3a_3x3'] = net

                # 147 x 147 x 64
                net = slim.conv2d(net, 120, 5, scope='Conv2d_1b_3x3')
                end_points['Conv2d_2b_3x3'] = net

                with tf.variable_scope('Logits'):
                    net = slim.flatten(net)
                    net = slim.fully_connected(net, 1024, scope='fc1', reuse=False)

                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='Dropout')

                    end_points['PreLogitsFlatten'] = net

                net = slim.fully_connected(net, bottleneck_layer_size, scope='Bottleneck', reuse=False)

    return net



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


class SiameseNet(object):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()

        input1_hold = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input1')
        input2_hold = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input2')

        with tf.variable_scope('siamese') as scope:
            self.embedding_net1 = embedding_net(input1_hold)
            scope.reuse_variables()
            self.embedding_net2 = embedding_net(input2_hold)

"""
class ClassificationNet(object):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        weight_decay = 5e-4
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.embeddings = tf.nn.l2_normalize(self.embedding_net, 1, 1e-10, name='embeddings')

        logits = slim.fully_connected(self.embedding_net, n_classes, activation_fn=None,
                                      weights_initializer=slim.initializers.xavier_initializer(),
                                      weights_regularizer=slim.l2_regularizer(weight_decay),
                                      scope='Logits', reuse=False)

        # Norm for the prelogits
        eps = 1e-4
        self.prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(self.embedding_net) + eps, ord=prelogits_norm_p, axis=1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.prelogits_norm * prelogits_norm_loss_factor)

        self.learning_rate = tf.train.exponential_decay(self.learning_rate_placeholder, self.global_step,
                                                        learning_rate_decay_epochs * self.epoch_size,
                                                        learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', self.learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.label_batch, logits=logits, name='cross_entropy_per_example')
        self.cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', self.cross_entropy_mean)

        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(self.label_batch, tf.int64)),
                                     tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

        # Calculate the total losses
        self.regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.total_loss = tf.add_n([self.cross_entropy_mean] + self.regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        self.train_op = facenet.train(self.total_loss, self.global_step, optimizer,
                                      self.learning_rate, moving_average_decay, tf.global_variables(),
                                      log_histograms)

class EmbeddingNet(object):
    def __init__(self, input_shape):
        super(EmbeddingNet, self).__init__()

        self.global_step = tf.Variable(0, trainable=False)
        self.input_hold = tf.placeholder(tf.float32, [None, input_shape[1]], name='l_input')
        # self.y_hold = tf.placeholder(tf.float32, [None, num_classes], name='y_hold')
        self.verif_labels_hold = tf.placeholder(tf.int32, shape=[None, ], name='verif_labels')
        # self.verif_labels_hold = tf.placeholder(tf.int32, shape=[None, ], name='verif_labels')
        self.keep_prob_hold = tf.placeholder(tf.float32, name='keep_prob')
        self.is_training_hold = tf.placeholder(tf.bool, name='is_training')
        self.learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        self.build(learning_rate=0.01)

    def build(self, learning_rate, global_step=None):
        fc1 = fc(self.input_hold, num_in, num_out, name='fc1')
        fc2 = fc(fc1, num_out, num_out, name='fc2')
        fc3 = fc(fc2, num_out, 512, name='fc3')
        fc4 = fc(fc3, 512, 80, name='fc4')
        #fc3 = fc(fc2, 160, 80, name='fc3')
        #fc4 = fc(fc3, 80, 80, name='fc4')

        # fc_1 = fc(concat_local_fc, 160, 4800, name='fc_1')
        # fc_1_dropout = tf.cond(self.is_training_hold, lambda: dropout(fc_1, self.keep_prob_hold), lambda: fc_1)

        output_fc = fc(fc4, 80, 2, relu=False, name='output')
        # output_fc = tf.nn.sigmoid(output_fc)  # TODO：是sigmoid激活还是relu激活，这个应该通过fc接口传递进去

        self.epoch_size = 1000
        learning_rate_decay_epochs = 100
        learning_rate_decay_factor = 0.9
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_placeholder, self.global_step,
                                                        learning_rate_decay_epochs * self.epoch_size,
                                                        learning_rate_decay_factor, staircase=True)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.verif_labels_hold, logits=output_fc, name='cross_entropy_per_example')
        self.cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        # tf.add_to_collection('losses', self.cross_entropy_mean)

        correct_prediction = tf.cast(tf.equal(tf.argmax(output_fc, 1), tf.cast(self.verif_labels_hold, tf.int64)),
                                     tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        moving_average_decay = 0.9999
        log_histograms = False
        self.train_op = facenet.train(self.cross_entropy_mean, self.global_step, 'ADAM', self.learning_rate,
                                      moving_average_decay, tf.global_variables(), log_histograms)
"""

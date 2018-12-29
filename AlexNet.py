
import numpy as np
import tensorflow as tf


def batch_norm(inputs, is_training, is_conv_out=True, decay=0.999):
    pass

with tf.device('/cpu:0'):
    learning_rate = 1e-4
    training_iters = 200
    batch_size = 200
    display_step = 5
    n_classes = 2
    n_fc1 = 4096
    n_fc2 = 2048

    x = tf.placeholder(tf.float32)




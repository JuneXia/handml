# He初始化，ELU激活函数
# Batch Normalization, 批量标准化
# 梯度裁剪
from functools import partial
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

datapath = "/home/xiajun/res/MNIST_data"
mnist = input_data.read_data_sets(datapath, validation_size=0, one_hot=True)


if __name__ == '__main__':
    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    batch_norm_momentum = 0.9
    learning_rate = 0.01

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name = 'X')
    y = tf.placeholder(tf.int64, shape=None, name = 'y')
    training = tf.placeholder_with_default(False, shape=(), name = 'training')  # 给Batch norm加一个placeholder

    with tf.name_scope("dnn"):
        he_init = tf.contrib.layers.variance_scaling_initializer()
        #对权重的初始化

        my_batch_norm_layer = partial(
            tf.layers.batch_normalization,
            training=training,
            momentum=batch_norm_momentum
        )

        my_dense_layer = partial(
            tf.layers.dense,
            kernel_initializer=he_init
        )

        hidden1 = my_dense_layer(X, n_hidden1, name='hidden1')
        bn1 = tf.nn.elu(my_batch_norm_layer(hidden1))
        hidden2 = my_dense_layer(bn1, n_hidden2, name='hidden2')
        bn2 = tf.nn.elu(my_batch_norm_layer(hidden2))
        logists_before_bn = my_dense_layer(bn2, n_outputs, name='outputs')
        logists = my_batch_norm_layer(logists_before_bn)

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logists)
        loss = tf.reduce_mean(xentropy, name='loss')

    # 训练操作1, ok
    '''
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
    '''

    # 训练操作2, ok
    ''
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            training_op = optimizer.minimize(loss)
    ''

    # 梯度裁剪, ok
    '''
    threshold = 1.0
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var) for grad, var in grads_and_vars]  # 将梯度裁剪到 -1.0 和 1.0 之间
    training_op = optimizer.apply_gradients(capped_gvs)
    '''

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logists, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epoches = 20
    batch_size = 200
# 注意：由于我们使用的是 tf.layers.batch_normalization() 而不是 tf.contrib.layers.batch_norm()（如本书所述），
# 所以我们需要明确运行批量规范化所需的额外更新操作（sess.run([ training_op，extra_update_ops], ...)。
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epoches):
            for iteraton in range(mnist.train.num_examples//batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                y_batch = np.argmax(y_batch, 1)

                # 训练操作1 用下面的sess.run
                # sess.run([training_op, extra_update_ops], feed_dict={training: True, X: X_batch, y: y_batch})

                # 训练操作2 用下面的sess.run
                sess.run(training_op, feed_dict={training: True, X: X_batch, y: y_batch})

            y_batch = np.argmax(mnist.test.labels, 1)
            accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images, y: y_batch})
            print(epoch, 'Test accuracy:', accuracy_val)
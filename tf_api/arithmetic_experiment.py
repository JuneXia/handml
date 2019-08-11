# 各种算术运算实验

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__1':
    # ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8],dtype = tf.int32)
    # indices = tf.constant([4, 3, 1, 7],dtype = tf.int32)
    # updates = tf.constant([9, 10, 11, 12],dtype = tf.int32)

    ref = tf.Variable([1, 2, 3, 4], dtype=tf.int32)
    indices = tf.constant([1, 3], dtype=tf.int32)
    updates = tf.constant([1, 3], dtype=tf.int32)

    sub = tf.scatter_sub(ref, indices, updates)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        arr = sess.run(sub)
        print(arr)


if __name__ == '__main__':
    learning_rate = 0.1
    decay_rate = 0.6
    global_steps = 1000
    decay_steps = 100

    # global_ = tf.Variable(tf.constant(0))
    global_ = tf.Variable(0)
    c = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=True)
    d = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=False)

    T_C = []
    F_D = []

    with tf.Session() as sess:
        for i in range(global_steps):
            T_c = sess.run(c, feed_dict={global_: i})
            T_C.append(T_c)
            F_d = sess.run(d, feed_dict={global_: i})
            F_D.append(F_d)

    plt.figure(1)
    plt.plot(range(global_steps), F_D, 'r-')
    plt.plot(range(global_steps), T_C, 'b-')

    plt.show()

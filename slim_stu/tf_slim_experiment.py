"""
# 程序19-1： 模型变量
import tensorflow.contrib.slim as slim
import tensorflow as tf

weight1 = slim.model_variable('weight1',
                            shape=[2, 3],
                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                            regularizer=slim.l2_regularizer(0.05))

weight2 = slim.model_variable('weight2',
                            shape=[2, 3],
                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                            regularizer=slim.l2_regularizer(0.05))

model_variables = slim.get_model_variables()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(weight1))
    print("--------------------")
    print(sess.run(model_variables))
    print("--------------------")
    print(sess.run(slim.get_variables_by_suffix("weight1")))
"""


"""
# 程序19-2：普通变量
import tensorflow.contrib.slim as slim
import tensorflow as tf

weight1 = slim.variable('weight1',
                            shape=[2, 3],
                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                            regularizer=slim.l2_regularizer(0.05)
                            )

weight2 = slim.variable('weight2',
                            shape=[2, 3],
                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                            regularizer=slim.l2_regularizer(0.05)
                            )

variables = slim.get_variables()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(weight1))
    print("--------------------")
    print(sess.run(variables))
"""


"""
# 程序19-3：slim管理用户自定义的变量
import tensorflow.contrib.slim as slim
import tensorflow as tf

weight = tf.Variable(tf.ones([2,3]))
slim.add_model_variable(weight)
model_variables = slim.get_model_variables()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(model_variables))
"""


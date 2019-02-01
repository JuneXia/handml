import os
import datetime
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

datapath = "/home/xiajun/res/MNIST_data"
mnist = input_data.read_data_sets(datapath, validation_size=0, one_hot=True)
model_field = "mnist_dnn5"
save_path = "./mnist_models/" + model_field + '_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
if os.path.exists(save_path) is not True:
    os.mkdir(save_path)

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 50
n_hidden3 = 50
n_hidden4 = 50
n_hidden5 = 50
n_outputs = 10

learning_rate = 0.01
n_epoches = 20
batch_size = 200

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=None, name='y')

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")

    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")
    hidden5 = tf.layers.dense(hidden4, n_hidden5, activation=tf.nn.relu, name="hidden5")

    logits = tf.layers.dense(hidden5, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

# 梯度裁剪
threshold = 1.0
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var) for grad, var in grads_and_vars]  # 将梯度裁剪到 -1.0 和 1.0 之间
training_op = optimizer.apply_gradients(capped_gvs)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoches):
        for iteraton in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            y_batch = np.argmax(y_batch, 1)

            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        y_batch = np.argmax(mnist.test.labels, 1)
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images, y: y_batch})
        print(epoch, 'Test accuracy:', accuracy_val)

    save_file = os.path.join(save_path, model_field + '.ckpt')
    save_path = saver.save(sess, save_file)

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

datapath = "/home/xiajun/res/MNIST_data"
mnist = input_data.read_data_sets(datapath, validation_size=0, one_hot=True)
pretrain_model = '/home/xiajun/dev/handml/mnist_models/mnist_dnn5_20181208182059/mnist_dnn5.ckpt'


n_inputs = 28 * 28   # MNIST
n_hidden1 = 300  # reused
n_hidden2 = 50   # reused
n_hidden3 = 50   # reused
n_hidden4 = 30   # new!
# n_hidden5 = 20   # new!
n_outputs = 10   # new!
# 预训练模型中有5个隐藏层，我们这里只有4个隐藏层，其中前3层是复用预训练模型的.

learning_rate = 0.01
n_epoches = 20
batch_size = 200

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
    # 原始模型中每个变量的名称应当和新模型中的名称保持完全相同
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")        # reused
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")  # reused
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")  # reused
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")  # new!
    # hidden5 = tf.layers.dense(hidden4, n_hidden5, activation=tf.nn.relu, name="hidden5")  # new!
    logits = tf.layers.dense(hidden4, n_outputs, name="outputs")                          # new!

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

    冻结较低层
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="hidden[34]|outputs")  # regular expression
    training_op = optimizer.minimize(loss, var_list=train_vars)

# build new model with the same definition as before for hidden layers 1-3
reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="hidden[123]")  # regular expression
reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
restore_saver = tf.train.Saver(reuse_vars_dict)  # to restore layers 1-3

init = tf.global_variables_initializer()
saver = tf.train.Saver()  # 用于保存新的模型

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, pretrain_model)

    for epoch in range(n_epoches):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            y_batch = np.argmax(y_batch, 1)

            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        y_batch = np.argmax(mnist.test.labels, 1)
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images, y: y_batch})
        print(epoch, "Test accuracy:", accuracy_val)

    save_path = saver.save(sess, "./mnist_models/new_model_reused_tmp.ckpt")
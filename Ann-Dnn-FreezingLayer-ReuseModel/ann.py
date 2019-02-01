# LTU
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
iris = load_iris()
X = iris.data[:, (2, 3)]  # 花瓣长度，宽度
y = (iris.target == 0).astype(np.int)
per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)
y_pred = per_clf.predict([[2, 0.5]])
print(y_pred)
"""

# MLP
"""
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train = iris.data
y_train = iris.target
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=3, feature_columns=feature_columns)
dnn_clf.fit(x=X_train, y=y_train, batch_size=50, steps=40)

y_predicted = list(dnn_clf.predict(X_train))
print(accuracy_score(y_train, y_predicted))
print(dnn_clf.evaluate(X_train, y_train))

print('end')
"""


# DNN
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

datapath = "/home/xiajun/res/MNIST_data"
mnist = input_data.read_data_sets(datapath, validation_size=0, one_hot=True)

n_epochs = 10
batch_size = 50
learning_rate = 0.01

n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None, n_outputs), name="y")
yy = tf.placeholder(tf.int64, shape=(None, ), name="yy")


def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X, W) + b
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z


# 创建DNN层方法1：
''
with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
    logits = neuron_layer(hidden2, n_outputs, "outputs")
''

# 创建DNN层方法2：
'''
from tensorflow.contrib.layers import fully_connected
with tf.name_scope("dnn"):
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
    logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)
'''

# 创建DNN层方法3：
'''
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

    #prediction = tf.layers.dense(hidden2, n_outputs)
    #logits = tf.identity(prediction, name='outputs')
'''


with tf.name_scope("loss"):
    # xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yy, logits=logits)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, 1), logits=logits)  # [3]
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, yy, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        yy_batch = np.argmax(y_batch, 1)
        acc_train = accuracy.eval(feed_dict={X: X_batch, yy: yy_batch})
        yy_batch = np.argmax(mnist.test.labels, 1)
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, yy: yy_batch})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./my_model_final.ckpt")










print('end')



# [1] hands on ML
# [2] TensorFlow实战实例》的一些错误更正ValueError: Only call `sparse_softmax_cross_entropy_with_logits` with named a  (https://blog.csdn.net/accumulate_zhang/article/details/78254417)

import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt


n_train = 20
n_test = 100
num_inputs = 200

true_w = np.ones((num_inputs, 1)) * 0.01
true_b = 0.05
features = np.random.normal(size=(n_train+n_test, num_inputs))
np.random.shuffle(features)
labels = np.dot(features, true_w) + true_b
labels += np.random.normal(scale=0.01, size=labels.shape)

train_features, test_features = features[:n_train], features[n_train:]
train_labels, test_labels = labels[:n_train], labels[n_train:]



batch_size = 1
epochs = 100
learning_rate = 0.003
lambd = 5



x = tf.placeholder(tf.float32, shape=(None, num_inputs))
y = tf.placeholder(tf.float32, shape=(None, 1))

w = tf.Variable(tf.random_normal((num_inputs, 1)))
b = tf.Variable(tf.zeros(1))
y_hat = tf.add(tf.matmul(x, w), b)



loss = tf.reduce_mean(tf.square(y-y_hat)) + lambd * (tf.reduce_sum(tf.pow(w, 2)) / 2)
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)



dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
dataset = dataset.repeat().batch(batch_size)
iterator = dataset.make_initializable_iterator()
next_data = iterator.get_next()



train_loss = []
test_loss = []
init = [tf.global_variables_initializer(), iterator.initializer]
with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        for _ in range(20):
            batch_x, batch_y = sess.run(next_data)
            sess.run(train, feed_dict={
                x: batch_x,
                y: batch_y
            })
        train_loss.append(sess.run(loss, feed_dict={
            x: train_features,
            y: train_labels
        }))
        test_loss.append(sess.run(loss, feed_dict={
            x: test_features,
            y: test_labels
        }))



plt.semilogy(range(1, epochs+1), train_loss)
plt.semilogy(range(1, epochs+1), test_loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

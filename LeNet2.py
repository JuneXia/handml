import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

datapath = '/home/xiajun/res/MNIST_data'
mnist_data_set = input_data.read_data_sets(datapath, validation_size=0, one_hot=True)

with tf.device('/gpu:0'):
    learning_rate = 1e-4
    batch_size = 200
    n_features = 28*28*1
    n_classes = 10
    n_fc1 = 7*7*120
    n_fc2 = 1024

    X = tf.placeholder(tf.float32, [None, n_features])
    y = tf.placeholder(tf.float32, [None, n_classes])

    W_conv = {
        'conv1': tf.Variable(tf.truncated_normal([5, 5, 1, 6], stddev=0.0001)),
        'conv2': tf.Variable(tf.truncated_normal([5, 5, 6, 16], stddev=0.01)),
        'conv3': tf.Variable(tf.truncated_normal([5, 5, 16, 120], stddev=0.01)),
        'fc1': tf.Variable(tf.truncated_normal([n_fc1, n_fc2], stddev=0.1)),
        'output': tf.Variable(tf.truncated_normal([n_fc2, n_classes], stddev=0.1))
    }
    b_conv = {
        'conv1': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[6])),
        'conv2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[16])),
        'conv3': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[120])),
        'fc1': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc2])),
        'output': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[n_classes]))
    }

    X_image = tf.reshape(X, [-1, 28, 28, 1])

    # 卷积层1
    conv1 = tf.nn.conv2d(X_image, W_conv['conv1'], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, b_conv['conv1'])
    # conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.sigmoid(conv1)

    # 池化层1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # LRN层
    # pool1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

    # 卷积层2
    conv2 = tf.nn.conv2d(pool1, W_conv['conv2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, b_conv['conv2'])
    # conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.sigmoid(conv2)
    # LRN层
    # conv2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

    # 池化层2
    # pool2 = tf.nn.avg_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    ## pool2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # 卷积层3
    conv3 = tf.nn.conv2d(pool2, W_conv['conv3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.bias_add(conv3, b_conv['conv3'])
    # conv3 = tf.nn.relu(conv3)
    conv3 = tf.nn.sigmoid(conv3)

    # 全连接层1
    reshape = tf.reshape(conv3, [-1, n_fc1])
    fc1 = tf.add(tf.matmul(reshape, W_conv['fc1']), b_conv['fc1'])
    # fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.sigmoid(fc1)

    # 输出层
    output = tf.add(tf.matmul(fc1, W_conv['output']), b_conv['output'])
    y_output = tf.nn.softmax(output)

    # 损失函数
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc3, y))  # failed
    cross_entropy = -tf.reduce_sum(y * tf.log(y_output))
    # 使用GD优化算法来调整参数
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # 评估模型
    correct_pred = tf.equal(tf.argmax(y_output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()


'''
# Debug
batch_xs, batch_ys = mnist_data_set.train.next_batch(5)
#x = tf.Variable(tf.truncated_normal([5, 5, 1, 6]))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(conv1.eval(feed_dict={X: batch_xs}).shape)
    print(pool1.eval(feed_dict={X: batch_xs}).shape)
    print(norm1.eval(feed_dict={X: batch_xs}).shape)
    print(conv2.eval(feed_dict={X: batch_xs}).shape)
    print(norm2.eval(feed_dict={X: batch_xs}).shape)
    print(pool2.eval(feed_dict={X: batch_xs}).shape)
    print(reshape.eval(feed_dict={X: batch_xs}).shape)
    print(fc1.eval(feed_dict={X: batch_xs}).shape)
    print(fc2.eval(feed_dict={X: batch_xs}).shape)
    print(fc3.eval(feed_dict={X: batch_xs}).shape)
    print('debug')
'''


loss_buf = []
accuracy_buf = []
with tf.Session() as sess:
    sess.run(init)

    total_batch = mnist_data_set.train.num_examples//batch_size
    for i in range(5):
        for iteration in range(10):
            batch_xs, batch_ys = mnist_data_set.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={X: batch_xs, y: batch_ys})

        batch_xs, batch_ys = mnist_data_set.train.images, mnist_data_set.train.labels
        loss_val = sess.run(cross_entropy, feed_dict={X: batch_xs, y: batch_ys})
        batch_xs, batch_ys = mnist_data_set.test.images[0:100, :], mnist_data_set.test.labels[0:100, :]
        # test_accuracy = accuracy.eval(feed_dict={X: batch_xs, y: batch_ys})
        test_accuracy = sess.run(accuracy, feed_dict={X: batch_xs, y: batch_ys})
        loss_buf.append(loss_val)
        accuracy_buf.append(test_accuracy)
        print("step {}, loss {}, testing accuracy {}".format(i, loss_val, test_accuracy))

# 画出准确率曲线
accuracy_ndarray = np.array(accuracy_buf)
accuracy_size = np.arange(len(accuracy_ndarray))
plt.plot(accuracy_size, accuracy_ndarray, 'b+', label='accuracy')

loss_ndarray = np.array(loss_buf)
loss_size = np.arange(len(loss_ndarray))
plt.plot(loss_size, loss_ndarray, 'r*', label='loss')

plt.show()


with open('LeNet-input?x28x28x1-conv?x28x28x6_maxpool?x14x14x6-conv?x14x14x16_maxpool?x7x7x16-conv?x7x7x120-fc?x1024-output?x10-epoch2000.csv', 'w') as fid:
    for loss, acc in zip(loss_buf, accuracy_buf):
        strText = str(loss) + ',' + str(acc) + '\n'
        fid.write(strText)
fid.close()

print('end')

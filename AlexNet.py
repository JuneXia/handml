import numpy as np
import cv2
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

datapath = '/home/xiajun/res/MNIST_data'
mnist_data_set = input_data.read_data_sets(datapath, validation_size=0, one_hot=True)


img = tf.Variable([[[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]], [[[13,14,15],[16,17,18]], [[19,20,21],[22,23,24]]]], dtype=tf.float32)
#img = tf.Variable(np.ones([2,3,4,5]), dtype=tf.float32)
mean, var = tf.nn.moments(img, [0, 1])

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    img = sess.run(img)
    print('img.shape = ', img.shape, '\n', img)

    mean = sess.run(mean)
    print('mean.shape = ', mean.shape, '\n', mean)

    var = sess.run(var)
    print('var.shape = ', var.shape, '\n', var)

    print('debug')



def batch_norm(inputs, is_training, is_conv_out=True, decay=0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])

        train_mean = tf.assign(pop_mean, pop_mean*decay+batch_mean*(1-decay))
        train_var = tf.assign(pop_var, pop_var*decay+batch_var*(1-decay))

        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, 0.001)


def image_shape_scale(batch_xs):
    images = np.reshape(batch_xs, [batch_xs.shape[0], 28, 28])
    imlist = []
    [imlist.append(cv2.resize(img, (227, 227))) for img in images]
    images = np.array(imlist)
    # cv2.imwrite('scale1.jpg', images[0]*200)
    # cv2.imwrite('scale2.jpg', images[1]*200)
    batch_xs = np.reshape(images, [batch_xs.shape[0], 227 * 227 * input_image_channel])
    return batch_xs

with tf.device('/cpu:0'):
    # 模型参数
    input_image_channel = 1
    learning_rate = 1e-4
    training_epoch = 200
    batch_size = 1
    n_classes = 10
    n_fc1 = 4096
    n_fc2 = 2048

    # 构建模型
    X = tf.placeholder(tf.float32, [None, 227 * 227 * input_image_channel])
    y = tf.placeholder(tf.float32, [None, n_classes])

    W_conv = {
        'conv1': tf.Variable(tf.truncated_normal([11, 11, input_image_channel, 96], stddev=0.0001)),
        'conv2': tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=0.01)),
        'conv3': tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.01)),
        'conv4': tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.01)),
        'conv5': tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.01)),
        'fc1': tf.Variable(tf.truncated_normal([6*6*256, n_fc1], stddev=0.1)),
        'fc2': tf.Variable(tf.truncated_normal([n_fc1, n_fc2], stddev=0.1)),
        'fc3': tf.Variable(tf.truncated_normal([n_fc2, n_classes], stddev=0.1))
    }

    b_conv = {
        'conv1': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[96])),
        'conv2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256])),
        'conv3': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384])),
        'conv4': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384])),
        'conv5': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256])),
        'fc1': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc1])),
        'fc2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc2])),
        'fc3': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[n_classes]))
    }

    X_image = tf.reshape(X, [-1, 227, 227, input_image_channel])

    # 卷积层1
    conv1 = tf.nn.conv2d(X_image, W_conv['conv1'], strides=[1, 4, 4, 1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, b_conv['conv1'])
    conv1 = batch_norm(conv1, True)
    conv1 = tf.nn.relu(conv1)
    # 此时 conv1.shape = [-1, 55, 55, 96]

    # 池化层1
    pool1 = tf.nn.avg_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    # pool1.shape = [-1, 27, 27, 96]
    # LRN层
    norm1 = tf.nn.lrn(pool1, 5, bias=1.0, alpha=0.001/9.0, beta=0.75)

    # 卷积层2
    conv2 = tf.nn.conv2d(norm1, W_conv['conv2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, b_conv['conv2'])
    conv2 = batch_norm(conv2, True)
    conv2 = tf.nn.relu(conv2)
    # 此时 conv2.shape = [-1, 27, 27, 256]

    # 池化层2
    pool2 = tf.nn.avg_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    # 此时 pool2.shape = [-1, 13, 13, 256]
    # LRN层
    norm2 = tf.nn.lrn(pool2, 5, bias=1.0, alpha=0.001/9.0, beta=0.75)

    # 卷积层3
    conv3 = tf.nn.conv2d(norm2, W_conv['conv3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.bias_add(conv3, b_conv['conv3'])
    conv3 = batch_norm(conv3, True)
    conv3 = tf.nn.relu(conv3)
    # 此时 conv3.shape = [-1, 13, 13, 384]

    # 卷积层4
    conv4 = tf.nn.conv2d(conv3, W_conv['conv4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = tf.nn.bias_add(conv4, b_conv['conv4'])
    conv4 = batch_norm(conv4, True)
    conv4 = tf.nn.relu(conv4)
    # 此时 conv4.shape = [-1, 13, 13, 384]

    # 卷积层5
    conv5 = tf.nn.conv2d(conv4, W_conv['conv5'], strides=[1, 1, 1, 1],padding='SAME')
    conv5 = tf.nn.bias_add(conv5, b_conv['conv5'])
    conv5 = batch_norm(conv5, True)
    conv5 = tf.nn.relu(conv5)
    # 此时 conv5.shape = [-1, 13, 13, 256]

    # 池化层5
    pool5 = tf.nn.avg_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    # 此时pool5.shape = [-1, 6, 6, 256]

    reshape = tf.reshape(pool5, [-1, 6*6*256])

    # 全连接层1
    fc1 = tf.add(tf.matmul(reshape, W_conv['fc1']), b_conv['fc1'])
    fc1 = batch_norm(fc1, True)
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, 0.5)

    # 全连接层2
    fc2 = tf.add(tf.matmul(fc1, W_conv['fc2']), b_conv['fc2'])
    fc2 = batch_norm(fc2, True)
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, 0.5)

    # 全连接层3
    fc3 = tf.add(tf.matmul(fc2, W_conv['fc3']), b_conv['fc3'])

    # 定义损失
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc3, y))
    cross_entropy = -tf.reduce_sum(y * tf.log(fc3))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # 评估模型
    correct_pred = tf.equal(tf.argmax(fc3, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    total_batch = mnist_data_set.train.num_examples // batch_size
    for i in range(training_epoch):
        for iteration in range(total_batch):
            batch_xs, batch_ys = mnist_data_set.train.next_batch(batch_size)
            batch_xs = image_shape_scale(batch_xs)
            sess.run(train_step, feed_dict={X: batch_xs, y: batch_ys})

        batch_xs, batch_ys = mnist_data_set.test.images, mnist_data_set.test.labels
        batch_xs = image_shape_scale(batch_xs)
        # train_accuracy = accuracy.eval(feed_dict={X: batch_xs, y: batch_ys})
        train_accuracy = sess.run(accuracy, feed_dict={X: batch_xs, y: batch_ys})
        print("step %d, training accuracy %g" % (i, train_accuracy))



import numpy as np
import cv2
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

datapath = '/home/xiajun/res/MNIST_data'
mnist_data_set = input_data.read_data_sets(datapath, validation_size=0, one_hot=True)


def image_shape_scale(batch_xs):
    images = np.reshape(batch_xs, [batch_xs.shape[0], 28, 28])
    imlist = []
    [imlist.append(cv2.resize(img, (227, 227))) for img in images]
    images = np.array(imlist)
    # cv2.imwrite('scale1.jpg', images[0]*200)
    # cv2.imwrite('scale2.jpg', images[1]*200)
    # batch_xs = np.reshape(images, [batch_xs.shape[0], 227 * 227 * input_image_channel])
    batch_xs = np.reshape(images, [batch_xs.shape[0], 227, 227, input_image_channel])
    return batch_xs


input_image_channel = 1
learning_rate = 1e-4
training_epoch = 50
batch_size = 200
n_classes = 10
n_fc1 = 6*6*256
n_fc2 = 4096
n_fc3 = 4096
dropout_rate = 0.5

X = tf.placeholder(tf.float32, [None, 227, 227, input_image_channel])
y = tf.placeholder(tf.float32, [None, n_classes])
learning_rate_holder = tf.placeholder(tf.float32)

W_conv = {
    'conv1': tf.Variable(tf.truncated_normal([11, 11, input_image_channel, 96], mean=0, stddev=0.01)),
    'conv2': tf.Variable(tf.truncated_normal([5, 5, 96, 256], mean=0, stddev=0.01)),
    'conv3': tf.Variable(tf.truncated_normal([3, 3, 256, 384], mean=0, stddev=0.01)),
    'conv4': tf.Variable(tf.truncated_normal([3, 3, 384, 384], mean=0, stddev=0.01)),
    'conv5': tf.Variable(tf.truncated_normal([3, 3, 384, 256], mean=0, stddev=0.01)),
    'fc1': tf.Variable(tf.truncated_normal([n_fc1, n_fc2], mean=0, stddev=0.01)),
    'fc2': tf.Variable(tf.truncated_normal([n_fc2, n_fc3], mean=0, stddev=0.01)),
    'output': tf.Variable(tf.truncated_normal([n_fc3, n_classes], mean=0, stddev=0.01))
}

b_conv = {
    'conv1': tf.Variable(tf.truncated_normal([96], mean=0.005, stddev=0.1)),
    'conv2': tf.Variable(tf.truncated_normal([256], mean=0.005, stddev=0.1)),
    'conv3': tf.Variable(tf.truncated_normal([384], mean=0.005, stddev=0.1)),
    'conv4': tf.Variable(tf.truncated_normal([384], mean=0.005, stddev=0.1)),
    'conv5': tf.Variable(tf.truncated_normal([256], mean=0.005, stddev=0.1)),
    'fc1': tf.Variable(tf.truncated_normal([n_fc2], mean=0.005, stddev=0.1)),
    'fc2': tf.Variable(tf.truncated_normal([n_fc3], mean=0.005, stddev=0.1)),
    'output': tf.Variable(tf.truncated_normal([n_classes], mean=0.005, stddev=0.1))
}

X_image = tf.reshape(X, [-1, 227, 227, input_image_channel])

# 卷积层1
conv1 = tf.nn.conv2d(X_image, W_conv['conv1'], strides=[1, 4, 4, 1], padding='VALID')
conv1 = tf.nn.bias_add(conv1, b_conv['conv1'])
conv1 = tf.nn.relu(conv1)
conv1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
# 此时 conv1.shape = [-1, 55, 55, 96]

# 池化层1
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
# pool1.shape = [-1, 27, 27, 96]

# 卷积层2
conv2 = tf.nn.conv2d(pool1, W_conv['conv2'], strides=[1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.bias_add(conv2, b_conv['conv2'])
conv2 = tf.nn.relu(conv2)
conv2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
# 此时 conv2.shape = [-1, 27, 27, 256]

# 池化层2
pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
# 此时 pool2.shape = [-1, 13, 13, 256]

# 卷积层3
conv3 = tf.nn.conv2d(pool2, W_conv['conv3'], strides=[1, 1, 1, 1], padding='SAME')
conv3 = tf.nn.bias_add(conv3, b_conv['conv3'])
conv3 = tf.nn.relu(conv3)
# 此时 conv3.shape = [-1, 13, 13, 384]

# 卷积层4
conv4 = tf.nn.conv2d(conv3, W_conv['conv4'], strides=[1, 1, 1, 1], padding='SAME')
conv4 = tf.nn.bias_add(conv4, b_conv['conv4'])
conv4 = tf.nn.relu(conv4)
# 此时 conv4.shape = [-1, 13, 13, 384]

# 卷积层5
conv5 = tf.nn.conv2d(conv4, W_conv['conv5'], strides=[1, 1, 1, 1], padding='SAME')
conv5 = tf.nn.bias_add(conv5, b_conv['conv5'])
conv5 = tf.nn.relu(conv5)
# 此时 conv5.shape = [-1, 13, 13, 256]

# 池化层5
pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
# 此时pool5.shape = [-1, 6, 6, 256]

# 全连接层1
reshape = tf.reshape(pool5, [-1, n_fc1])
# 此时reshape.shape = [-1, 9216]
fc1 = tf.add(tf.matmul(reshape, W_conv['fc1']), b_conv['fc1'])
fc1 = tf.nn.relu(fc1)
fc1 = tf.nn.dropout(fc1, dropout_rate)
# 此时fc1.shape = [-1, 4096]

# 全连接层2
fc2 = tf.add(tf.matmul(fc1, W_conv['fc2']), b_conv['fc2'])
fc2 = tf.nn.relu(fc2)
fc2 = tf.nn.dropout(fc2, dropout_rate)
# 此时fc2.shape = [-1, 4096]

# 输出层
output = tf.add(tf.matmul(fc2, W_conv['output']), b_conv['output'])
# 此时output.shape = [-1. 10]

# 定义交叉熵损失函数（有两种方法）：
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 方法1： 自己实现交叉熵
y_output = tf.nn.softmax(output)  # 对网络最后一层的输出做softmax, 这通常是求取输出属于某一类的概率
cross_entropy = -tf.reduce_sum(y * tf.log(y_output))  # 用softmax的输出向量和样本的实际标签做一个交叉熵.
loss = tf.reduce_mean(cross_entropy)  # 对交叉熵求均值就是loss
# loss = -tf.reduce_mean(y * tf.log(y_output))  # 交叉熵本应是一个向量，但tf.reduce_mean可以直接求取tensor所有维度的和，所以这里可以用tf.reduce_mean一句代替上述三步。

# 方法2：使用tensorflow自带的tf.nn.softmax_cross_entropy_with_logits函数实现交叉熵
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_holder).minimize(loss)

# 评估模型
correct_pred = tf.equal(tf.argmax(y_output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

loss_buf = []
accuracy_buf = []
with tf.device("/gpu:0"):
    # with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=True)
    # config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        sess.run(init)

        total_batch = mnist_data_set.train.num_examples // batch_size
        for i in range(training_epoch):
            for iteration in range(total_batch):
                batch_xs, batch_ys = mnist_data_set.train.next_batch(batch_size)
                batch_xs = image_shape_scale(batch_xs)

                if i < 30:
                    sess.run(train_step, feed_dict={X: batch_xs, y: batch_ys, learning_rate_holder: learning_rate})
                elif i < 50:
                    sess.run(train_step, feed_dict={X: batch_xs, y: batch_ys, learning_rate_holder: learning_rate/10.0})
                elif i < 70:
                    sess.run(train_step, feed_dict={X: batch_xs, y: batch_ys, learning_rate_holder: learning_rate/100.0})
                else:
                    sess.run(train_step, feed_dict={X: batch_xs, y: batch_ys, learning_rate_holder: learning_rate/1000.0})
                test_accuracy = sess.run(accuracy, feed_dict={X: batch_xs, y: batch_ys})
                print("step {}, iteration {}, training accuracy {}".format(i, iteration, test_accuracy))

            batch_xs, batch_ys = mnist_data_set.test.images[0:1000, :], mnist_data_set.test.labels[0:1000, :]
            batch_xs = image_shape_scale(batch_xs)

            loss_val = sess.run(loss, feed_dict={X: batch_xs, y: batch_ys})
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

# 保存loss和测试准确率到csv文件
with open('AlexNet.csv', 'w') as fid:
    for loss, acc in zip(loss_buf, accuracy_buf):
        strText = str(loss) + ',' + str(acc) + '\n'
        fid.write(strText)
fid.close()

print('end')

'''
# AlexNet-input?x227x227x1-conv?x55x55x96lrn-maxpool?x27x27x96-conv?x27x27x256lrn-maxpool?x13x13x256-conv?x13x13x384-conv?x13x13x384-conv?x13x13x256-maxpool?x6x6x256-fc?x4096-fc?x4096-output?x10-learnrate_feed-customInit-epoch50.csv
动态改变学习率，消除梯度爆炸


step 1, loss 2300.436, testing accuracy 0.117
step 2, loss 2299.375, testing accuracy 0.13
... ...
step 10, loss 2290.4155, testing accuracy 0.154
step 11, loss 678.09644, testing accuracy 0.788
step 12, loss 178.35306, testing accuracy 0.948
... ...
step 49, loss 35.42096, testing accuracy 0.988
step 50, loss 34.458153, testing accuracy 0.988


'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

datapath = '/home/xiajun/res/MNIST_data'
mnist_data_set = input_data.read_data_sets(datapath, validation_size=0, one_hot=True)

x = tf.placeholder('float', [None, 784])
y_ = tf.placeholder('float', [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

#第一层卷积层，初始化卷积核参数、偏置值，该卷积层5*5大小，1个通道，共有6个不同卷积核
filter1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6]))
bias1 = tf.Variable(tf.truncated_normal([6]))
conv1 = tf.nn.conv2d(x_image, filter1, strides=[1, 1, 1, 1], padding='SAME')
# 此时conv1.shape = [-1, 28, 28, 6]
h_conv1 = tf.nn.sigmoid(conv1 + bias1)
# h_conv1 = tf.nn.relu(conv1 + bias1)

maxPool2 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 此时maxPool2.shape = [-1, 14, 14, 6]

filter2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16]))
bias2 = tf.Variable(tf.truncated_normal([16]))
conv2 = tf.nn.conv2d(maxPool2, filter2, strides=[1, 1, 1, 1], padding='SAME')
# 此时conv2.shape = [-1, 14, 14, 16]
h_conv2 = tf.nn.sigmoid(conv2 + bias2)
# h_conv2 = tf.nn.relu(conv2 + bias2)

maxPool3 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 此时maxPool3.shape = [-1, 7, 7, 16]

filter3 = tf.Variable(tf.truncated_normal([5, 5, 16, 120]))
bias3 = tf.Variable(tf.truncated_normal([120]))
conv3 = tf.nn.conv2d(maxPool3, filter3, strides=[1, 1, 1, 1], padding='SAME')
# 此时conv3.shape = [-1, 7, 7, 120]
h_conv3 = tf.nn.sigmoid(conv3 + bias3)
# h_conv3 = tf.nn.relu(conv3 + bias3)


# 全连接层
# 权值参数
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 120, 80]))
# 偏置值
b_fc1 = tf.Variable(tf.truncated_normal([80]))
# 将卷积的产出展开
h_pool2_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 120])
# 神经网络计算，并添加sigmoid激活函数
h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# 此时h_fc1.shape = [-1, 80]

# 输出层，使用softmax进行多分类
W_fc2 = tf.Variable(tf.truncated_normal([80, 10]))
b_fc2 = tf.Variable(tf.truncated_normal([10]))
y_output = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

# 损失函数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_output))
# 使用GD优化算法来调整参数
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

sess = tf.InteractiveSession()

# 测试正确率
correct_prediction = tf.equal(tf.argmax(y_output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 所有变量进行初始化
sess.run(tf.initialize_all_variables())



'''
# Debug
batch_xs, batch_ys = mnist_data_set.train.next_batch(5)
# x = tf.Variable(tf.truncated_normal([5, 5, 1, 6]))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(conv1.eval(feed_dict={x: batch_xs}).shape)
    print(h_conv1.eval(feed_dict={x: batch_xs}).shape)
    print(maxPool2.eval(feed_dict={x: batch_xs}).shape)
    print(conv2.eval(feed_dict={x: batch_xs}).shape)
    print(h_conv2.eval(feed_dict={x: batch_xs}).shape)
    print(maxPool3.eval(feed_dict={x: batch_xs}).shape)
    print(h_conv3.eval(feed_dict={x: batch_xs}).shape)
    print(h_fc1.eval(feed_dict={x: batch_xs}).shape)
    print('debug')
'''


# 进行训练
batch_size = 200
start_time = time.time()
for i in range(20000):
    for iteration in range(mnist_data_set.train.num_examples//batch_size):
        # 获取训练数据
        batch_xs, batch_ys = mnist_data_set.train.next_batch(batch_size)
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys})

    batch_xs, batch_ys = mnist_data_set.test.images, mnist_data_set.test.labels
    train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
    print("step %d, training accuracy %g" % (i, train_accuracy))

    end_time = time.time()
    print('time: ', (end_time - start_time))
    start_time = end_time

# 关闭会话
sess.close()


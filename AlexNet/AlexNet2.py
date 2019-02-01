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


train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

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

                sess.run(train_step, feed_dict={X: batch_xs, y: batch_ys})
                test_accuracy = sess.run(accuracy, feed_dict={X: batch_xs, y: batch_ys})
                if iteration % 4 == 0:
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
# AlexNet-input?x227x227x1-conv?x55x55x96lrn-maxpool?x27x27x96-conv?x27x27x256lrn-maxpool?x13x13x256-conv?x13x13x384-conv?x13x13x384-conv?x13x13x256-maxpool?x6x6x256-fc?x4096-fc?x4096-output?x10-learnrate_1e-4-customInit-epoch50.csv
刚开始会收敛，中途会突然梯度爆炸
'''



'''
step 1, loss 2299.4792, testing accuracy 0.122
step 2, loss 2299.8113, testing accuracy 0.117
step 3, loss 2301.2236, testing accuracy 0.124
step 4, loss 2299.0164, testing accuracy 0.127
step 5, loss 2299.6777, testing accuracy 0.116
step 6, loss 2300.289, testing accuracy 0.113
step 7, loss 2301.5657, testing accuracy 0.11
step 8, loss 2302.0884, testing accuracy 0.121
step 9, loss 2297.6191, testing accuracy 0.121
step 10, loss 2296.3777, testing accuracy 0.119
step 11, loss 2293.2812, testing accuracy 0.125
step 12, loss 1955.1497, testing accuracy 0.391
step 13, loss 353.29596, testing accuracy 0.866
step 14, loss 146.65419, testing accuracy 0.954

... ...

step 44, loss 36.49175, testing accuracy 0.99
step 45, loss 22.933325, testing accuracy 0.986
step 46, loss 35.3011, testing accuracy 0.99
step 47, loss nan, testing accuracy 0.085
step 48, loss nan, testing accuracy 0.085
step 49, loss nan, testing accuracy 0.085
step 50, loss nan, testing accuracy 0.085









step 45, loss 35.30110168457031, testing accuracy 0.9900000095367432
step 46, iteration 0, training accuracy 0.9950000047683716
step 46, iteration 1, training accuracy 1.0
step 46, iteration 2, training accuracy 1.0
step 46, iteration 3, training accuracy 1.0
step 46, iteration 4, training accuracy 1.0
step 46, iteration 5, training accuracy 1.0
step 46, iteration 6, training accuracy 1.0
step 46, iteration 7, training accuracy 1.0
step 46, iteration 8, training accuracy 1.0
step 46, iteration 9, training accuracy 1.0
step 46, iteration 10, training accuracy 1.0
step 46, iteration 11, training accuracy 1.0
step 46, iteration 12, training accuracy 1.0
step 46, iteration 13, training accuracy 1.0
step 46, iteration 14, training accuracy 0.9950000047683716
step 46, iteration 15, training accuracy 1.0
step 46, iteration 16, training accuracy 1.0
step 46, iteration 17, training accuracy 1.0
step 46, iteration 18, training accuracy 0.9950000047683716
step 46, iteration 19, training accuracy 0.9900000095367432
step 46, iteration 20, training accuracy 1.0
step 46, iteration 21, training accuracy 1.0
step 46, iteration 22, training accuracy 0.9950000047683716
step 46, iteration 23, training accuracy 1.0
step 46, iteration 24, training accuracy 1.0
step 46, iteration 25, training accuracy 0.10499999672174454
step 46, iteration 26, training accuracy 0.09000000357627869
step 46, iteration 27, training accuracy 0.07999999821186066
step 46, iteration 28, training accuracy 0.07000000029802322
step 46, iteration 29, training accuracy 0.07999999821186066
step 46, iteration 30, training accuracy 0.1550000011920929
step 46, iteration 31, training accuracy 0.10000000149011612
step 46, iteration 32, training accuracy 0.05000000074505806
step 46, iteration 33, training accuracy 0.07999999821186066
step 46, iteration 34, training accuracy 0.06499999761581421
step 46, iteration 35, training accuracy 0.10000000149011612
step 46, iteration 36, training accuracy 0.0949999988079071
step 46, iteration 37, training accuracy 0.11999999731779099
step 46, iteration 38, training accuracy 0.10499999672174454
step 46, iteration 39, training accuracy 0.08500000089406967
step 46, iteration 40, training accuracy 0.0949999988079071
step 46, iteration 41, training accuracy 0.10499999672174454
step 46, iteration 42, training accuracy 0.0949999988079071
step 46, iteration 43, training accuracy 0.0949999988079071
step 46, iteration 44, training accuracy 0.09000000357627869
step 46, iteration 45, training accuracy 0.10999999940395355
step 46, iteration 46, training accuracy 0.07000000029802322
step 46, iteration 47, training accuracy 0.12999999523162842
step 46, iteration 48, training accuracy 0.07500000298023224
step 46, iteration 49, training accuracy 0.11999999731779099
step 46, iteration 50, training accuracy 0.10999999940395355
step 46, iteration 51, training accuracy 0.10000000149011612
step 46, iteration 52, training accuracy 0.0949999988079071
step 46, iteration 53, training accuracy 0.08500000089406967
step 46, iteration 54, training accuracy 0.15000000596046448
step 46, iteration 55, training accuracy 0.10499999672174454
step 46, iteration 56, training accuracy 0.125
step 46, iteration 57, training accuracy 0.07000000029802322
step 46, iteration 58, training accuracy 0.15000000596046448
step 46, iteration 59, training accuracy 0.09000000357627869
step 46, iteration 60, training accuracy 0.08500000089406967
step 46, iteration 61, training accuracy 0.10000000149011612
step 46, iteration 62, training accuracy 0.14499999582767487
step 46, iteration 63, training accuracy 0.10999999940395355
step 46, iteration 64, training accuracy 0.0949999988079071
step 46, iteration 65, training accuracy 0.0949999988079071
step 46, iteration 66, training accuracy 0.11500000208616257
step 46, iteration 67, training accuracy 0.0949999988079071
step 46, iteration 68, training accuracy 0.09000000357627869
step 46, iteration 69, training accuracy 0.12999999523162842
step 46, iteration 70, training accuracy 0.0949999988079071
step 46, iteration 71, training accuracy 0.14499999582767487
step 46, iteration 72, training accuracy 0.0949999988079071
step 46, iteration 73, training accuracy 0.07999999821186066
step 46, iteration 74, training accuracy 0.10000000149011612
step 46, iteration 75, training accuracy 0.10000000149011612
step 46, iteration 76, training accuracy 0.07500000298023224
step 46, iteration 77, training accuracy 0.10000000149011612
step 46, iteration 78, training accuracy 0.125
step 46, iteration 79, training accuracy 0.10000000149011612
step 46, iteration 80, training accuracy 0.11999999731779099
step 46, iteration 81, training accuracy 0.054999999701976776
step 46, iteration 82, training accuracy 0.0949999988079071
step 46, iteration 83, training accuracy 0.10999999940395355
step 46, iteration 84, training accuracy 0.12999999523162842
step 46, iteration 85, training accuracy 0.08500000089406967
step 46, iteration 86, training accuracy 0.10499999672174454
step 46, iteration 87, training accuracy 0.07000000029802322
step 46, iteration 88, training accuracy 0.07500000298023224
step 46, iteration 89, training accuracy 0.07999999821186066
step 46, iteration 90, training accuracy 0.10000000149011612
step 46, iteration 91, training accuracy 0.07999999821186066
step 46, iteration 92, training accuracy 0.07999999821186066
step 46, iteration 93, training accuracy 0.08500000089406967
step 46, iteration 94, training accuracy 0.14000000059604645
step 46, iteration 95, training accuracy 0.10999999940395355
step 46, iteration 96, training accuracy 0.125
step 46, iteration 97, training accuracy 0.11500000208616257
step 46, iteration 98, training accuracy 0.10000000149011612
step 46, iteration 99, training accuracy 0.10999999940395355
step 46, iteration 100, training accuracy 0.05999999865889549
step 46, iteration 101, training accuracy 0.07000000029802322
step 46, iteration 102, training accuracy 0.11999999731779099
step 46, iteration 103, training accuracy 0.09000000357627869
step 46, iteration 104, training accuracy 0.04500000178813934
step 46, iteration 105, training accuracy 0.10999999940395355
step 46, iteration 106, training accuracy 0.09000000357627869
step 46, iteration 107, training accuracy 0.11500000208616257
step 46, iteration 108, training accuracy 0.09000000357627869
step 46, iteration 109, training accuracy 0.10000000149011612
step 46, iteration 110, training accuracy 0.09000000357627869
step 46, iteration 111, training accuracy 0.0949999988079071
step 46, iteration 112, training accuracy 0.10999999940395355
step 46, iteration 113, training accuracy 0.125
step 46, iteration 114, training accuracy 0.09000000357627869
step 46, iteration 115, training accuracy 0.10000000149011612
step 46, iteration 116, training accuracy 0.11500000208616257
step 46, iteration 117, training accuracy 0.11500000208616257
step 46, iteration 118, training accuracy 0.10499999672174454
step 46, iteration 119, training accuracy 0.07500000298023224
step 46, iteration 120, training accuracy 0.08500000089406967
step 46, iteration 121, training accuracy 0.125
step 46, iteration 122, training accuracy 0.07999999821186066
step 46, iteration 123, training accuracy 0.09000000357627869
step 46, iteration 124, training accuracy 0.07999999821186066
step 46, iteration 125, training accuracy 0.10000000149011612
step 46, iteration 126, training accuracy 0.09000000357627869
step 46, iteration 127, training accuracy 0.09000000357627869
step 46, iteration 128, training accuracy 0.11500000208616257
step 46, iteration 129, training accuracy 0.10499999672174454
step 46, iteration 130, training accuracy 0.08500000089406967
step 46, iteration 131, training accuracy 0.1599999964237213
step 46, iteration 132, training accuracy 0.10000000149011612
step 46, iteration 133, training accuracy 0.10000000149011612
step 46, iteration 134, training accuracy 0.0949999988079071
step 46, iteration 135, training accuracy 0.11500000208616257
step 46, iteration 136, training accuracy 0.07999999821186066
step 46, iteration 137, training accuracy 0.10499999672174454
step 46, iteration 138, training accuracy 0.07000000029802322
step 46, iteration 139, training accuracy 0.10499999672174454
step 46, iteration 140, training accuracy 0.14000000059604645
step 46, iteration 141, training accuracy 0.05000000074505806
step 46, iteration 142, training accuracy 0.07000000029802322
step 46, iteration 143, training accuracy 0.07500000298023224
step 46, iteration 144, training accuracy 0.09000000357627869
step 46, iteration 145, training accuracy 0.09000000357627869
step 46, iteration 146, training accuracy 0.11999999731779099
step 46, iteration 147, training accuracy 0.07000000029802322
step 46, iteration 148, training accuracy 0.0949999988079071
step 46, iteration 149, training accuracy 0.0949999988079071
step 46, iteration 150, training accuracy 0.10499999672174454
step 46, iteration 151, training accuracy 0.09000000357627869
step 46, iteration 152, training accuracy 0.10999999940395355
step 46, iteration 153, training accuracy 0.0949999988079071
step 46, iteration 154, training accuracy 0.0949999988079071
step 46, iteration 155, training accuracy 0.10499999672174454
step 46, iteration 156, training accuracy 0.11500000208616257
step 46, iteration 157, training accuracy 0.09000000357627869
step 46, iteration 158, training accuracy 0.0949999988079071
step 46, iteration 159, training accuracy 0.10000000149011612
step 46, iteration 160, training accuracy 0.09000000357627869
step 46, iteration 161, training accuracy 0.10999999940395355
step 46, iteration 162, training accuracy 0.10000000149011612
step 46, iteration 163, training accuracy 0.07000000029802322
step 46, iteration 164, training accuracy 0.07500000298023224
step 46, iteration 165, training accuracy 0.0949999988079071
step 46, iteration 166, training accuracy 0.10999999940395355
step 46, iteration 167, training accuracy 0.0949999988079071
step 46, iteration 168, training accuracy 0.07500000298023224
step 46, iteration 169, training accuracy 0.10499999672174454
step 46, iteration 170, training accuracy 0.07500000298023224
step 46, iteration 171, training accuracy 0.07999999821186066
step 46, iteration 172, training accuracy 0.11999999731779099
step 46, iteration 173, training accuracy 0.10000000149011612
step 46, iteration 174, training accuracy 0.10000000149011612
step 46, iteration 175, training accuracy 0.08500000089406967
step 46, iteration 176, training accuracy 0.13500000536441803
step 46, iteration 177, training accuracy 0.07999999821186066
step 46, iteration 178, training accuracy 0.11999999731779099
step 46, iteration 179, training accuracy 0.11999999731779099
step 46, iteration 180, training accuracy 0.11999999731779099
step 46, iteration 181, training accuracy 0.08500000089406967
step 46, iteration 182, training accuracy 0.10499999672174454
step 46, iteration 183, training accuracy 0.07999999821186066
step 46, iteration 184, training accuracy 0.11500000208616257
step 46, iteration 185, training accuracy 0.0949999988079071
step 46, iteration 186, training accuracy 0.07500000298023224
step 46, iteration 187, training accuracy 0.10000000149011612
step 46, iteration 188, training accuracy 0.08500000089406967
step 46, iteration 189, training accuracy 0.08500000089406967
step 46, iteration 190, training accuracy 0.10999999940395355
step 46, iteration 191, training accuracy 0.10499999672174454
step 46, iteration 192, training accuracy 0.13500000536441803
step 46, iteration 193, training accuracy 0.125
step 46, iteration 194, training accuracy 0.10000000149011612
step 46, iteration 195, training accuracy 0.10000000149011612
step 46, iteration 196, training accuracy 0.08500000089406967
step 46, iteration 197, training accuracy 0.11999999731779099
step 46, iteration 198, training accuracy 0.07000000029802322
step 46, iteration 199, training accuracy 0.08500000089406967
step 46, iteration 200, training accuracy 0.09000000357627869
step 46, iteration 201, training accuracy 0.11500000208616257
step 46, iteration 202, training accuracy 0.14000000059604645
step 46, iteration 203, training accuracy 0.0949999988079071
step 46, iteration 204, training accuracy 0.11500000208616257
step 46, iteration 205, training accuracy 0.10999999940395355
step 46, iteration 206, training accuracy 0.07999999821186066
step 46, iteration 207, training accuracy 0.10999999940395355
step 46, iteration 208, training accuracy 0.08500000089406967
step 46, iteration 209, training accuracy 0.0949999988079071
step 46, iteration 210, training accuracy 0.11500000208616257
step 46, iteration 211, training accuracy 0.09000000357627869
step 46, iteration 212, training accuracy 0.0949999988079071
step 46, iteration 213, training accuracy 0.0949999988079071
step 46, iteration 214, training accuracy 0.11500000208616257
step 46, iteration 215, training accuracy 0.0949999988079071
step 46, iteration 216, training accuracy 0.09000000357627869
step 46, iteration 217, training accuracy 0.10499999672174454
step 46, iteration 218, training accuracy 0.0949999988079071
step 46, iteration 219, training accuracy 0.10999999940395355
step 46, iteration 220, training accuracy 0.10499999672174454
step 46, iteration 221, training accuracy 0.10000000149011612
step 46, iteration 222, training accuracy 0.0949999988079071
step 46, iteration 223, training accuracy 0.07999999821186066
step 46, iteration 224, training accuracy 0.09000000357627869
step 46, iteration 225, training accuracy 0.10499999672174454
step 46, iteration 226, training accuracy 0.10000000149011612
step 46, iteration 227, training accuracy 0.0949999988079071
step 46, iteration 228, training accuracy 0.10499999672174454
step 46, iteration 229, training accuracy 0.0949999988079071
step 46, iteration 230, training accuracy 0.08500000089406967
step 46, iteration 231, training accuracy 0.10000000149011612
step 46, iteration 232, training accuracy 0.17000000178813934
step 46, iteration 233, training accuracy 0.10000000149011612
step 46, iteration 234, training accuracy 0.10499999672174454
step 46, iteration 235, training accuracy 0.09000000357627869
step 46, iteration 236, training accuracy 0.10499999672174454
step 46, iteration 237, training accuracy 0.07000000029802322
step 46, iteration 238, training accuracy 0.09000000357627869
step 46, iteration 239, training accuracy 0.10999999940395355
step 46, iteration 240, training accuracy 0.08500000089406967
step 46, iteration 241, training accuracy 0.0949999988079071
step 46, iteration 242, training accuracy 0.08500000089406967
step 46, iteration 243, training accuracy 0.13500000536441803
step 46, iteration 244, training accuracy 0.06499999761581421
step 46, iteration 245, training accuracy 0.0949999988079071
step 46, iteration 246, training accuracy 0.09000000357627869
step 46, iteration 247, training accuracy 0.10000000149011612
step 46, iteration 248, training accuracy 0.10000000149011612
step 46, iteration 249, training accuracy 0.10499999672174454
step 46, iteration 250, training accuracy 0.125
step 46, iteration 251, training accuracy 0.0949999988079071
step 46, iteration 252, training accuracy 0.0949999988079071
step 46, iteration 253, training accuracy 0.10000000149011612
step 46, iteration 254, training accuracy 0.10000000149011612
step 46, iteration 255, training accuracy 0.10000000149011612
step 46, iteration 256, training accuracy 0.09000000357627869
step 46, iteration 257, training accuracy 0.10000000149011612
step 46, iteration 258, training accuracy 0.10000000149011612
step 46, iteration 259, training accuracy 0.09000000357627869
step 46, iteration 260, training accuracy 0.10999999940395355
step 46, iteration 261, training accuracy 0.07500000298023224
step 46, iteration 262, training accuracy 0.0949999988079071
step 46, iteration 263, training accuracy 0.06499999761581421
step 46, iteration 264, training accuracy 0.0949999988079071
step 46, iteration 265, training accuracy 0.0949999988079071
step 46, iteration 266, training accuracy 0.11999999731779099
step 46, iteration 267, training accuracy 0.1550000011920929
step 46, iteration 268, training accuracy 0.06499999761581421
step 46, iteration 269, training accuracy 0.09000000357627869
step 46, iteration 270, training accuracy 0.09000000357627869
step 46, iteration 271, training accuracy 0.09000000357627869
step 46, iteration 272, training accuracy 0.07999999821186066
step 46, iteration 273, training accuracy 0.11500000208616257
step 46, iteration 274, training accuracy 0.09000000357627869
step 46, iteration 275, training accuracy 0.11999999731779099
step 46, iteration 276, training accuracy 0.07000000029802322
step 46, iteration 277, training accuracy 0.09000000357627869
step 46, iteration 278, training accuracy 0.10000000149011612
step 46, iteration 279, training accuracy 0.11500000208616257
step 46, iteration 280, training accuracy 0.07999999821186066
step 46, iteration 281, training accuracy 0.09000000357627869
step 46, iteration 282, training accuracy 0.04500000178813934
step 46, iteration 283, training accuracy 0.08500000089406967
step 46, iteration 284, training accuracy 0.13500000536441803
step 46, iteration 285, training accuracy 0.11500000208616257
step 46, iteration 286, training accuracy 0.09000000357627869
step 46, iteration 287, training accuracy 0.10000000149011612
step 46, iteration 288, training accuracy 0.07999999821186066
step 46, iteration 289, training accuracy 0.11500000208616257
step 46, iteration 290, training accuracy 0.14000000059604645
step 46, iteration 291, training accuracy 0.10000000149011612
step 46, iteration 292, training accuracy 0.11999999731779099
step 46, iteration 293, training accuracy 0.11999999731779099
step 46, iteration 294, training accuracy 0.05999999865889549
step 46, iteration 295, training accuracy 0.08500000089406967
step 46, iteration 296, training accuracy 0.08500000089406967
step 46, iteration 297, training accuracy 0.10000000149011612
step 46, iteration 298, training accuracy 0.06499999761581421
step 46, iteration 299, training accuracy 0.10000000149011612
step 46, loss nan, testing accuracy 0.08500000089406967
step 47, iteration 0, training accuracy 0.10499999672174454
step 47, iteration 1, training accuracy 0.06499999761581421
step 47, iteration 2, training accuracy 0.08500000089406967
step 47, iteration 3, training accuracy 0.05999999865889549
step 47, iteration 4, training accuracy 0.11500000208616257
step 47, iteration 5, training accuracy 0.10499999672174454
step 47, iteration 6, training accuracy 0.07500000298023224
step 47, iteration 7, training accuracy 0.10999999940395355
step 47, iteration 8, training accuracy 0.11999999731779099
step 47, iteration 9, training accuracy 0.12999999523162842
step 47, iteration 10, training accuracy 0.10499999672174454
step 47, iteration 11, training accuracy 0.09000000357627869
step 47, iteration 12, training accuracy 0.10499999672174454
step 47, iteration 13, training accuracy 0.125
step 47, iteration 14, training accuracy 0.11500000208616257
step 47, iteration 15, training accuracy 0.0949999988079071
step 47, iteration 16, training accuracy 0.10999999940395355
step 47, iteration 17, training accuracy 0.07500000298023224
step 47, iteration 18, training accuracy 0.11999999731779099
step 47, iteration 19, training accuracy 0.07999999821186066
step 47, iteration 20, training accuracy 0.08500000089406967
step 47, iteration 21, training accuracy 0.09000000357627869
step 47, iteration 22, training accuracy 0.07999999821186066
step 47, iteration 23, training accuracy 0.13500000536441803
step 47, iteration 24, training accuracy 0.0949999988079071
'''
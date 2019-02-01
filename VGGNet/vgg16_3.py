import numpy as np
import cv2
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

datapath = '/home/xiajun/res/MNIST_data'
mnist_data_set = input_data.read_data_sets(datapath, validation_size=0, one_hot=True)

num_classes = 10
learning_rate = 1e-4
training_epoch = 50
batch_size = 64
input_image_shape = (224, 224, 1)
conv_layer_trainable = True


def image_shape_scale(batch_xs, input_image_shape):
    images = np.reshape(batch_xs, [batch_xs.shape[0], 28, 28])
    imlist = []
    [imlist.append(cv2.resize(img, input_image_shape[0:2])) for img in images]
    images = np.array(imlist)
    # cv2.imwrite('scale1.jpg', images[0]*200)
    # cv2.imwrite('scale2.jpg', images[1]*200)
    # batch_xs = np.reshape(images, [batch_xs.shape[0], 227 * 227 * input_image_channel])
    batch_xs = np.reshape(images, [batch_xs.shape[0], input_image_shape[0], input_image_shape[1], input_image_shape[2]])
    return batch_xs


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


class vgg16:
    def __init__(self, imgs):
        self.parameters = []
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.probs = self.fc8

    def saver(self):
        return tf.train.Saver()

    def maxpool(self, name, input_data):
        out = tf.nn.max_pool(input_data, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name=name)
        return out

    def conv(self, name, input_data, out_channel):
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable('weights', [3, 3, in_channel, out_channel], dtype=tf.float32, trainable=conv_layer_trainable)
            biases = tf.get_variable('biases', [out_channel], dtype=tf.float32, trainable=conv_layer_trainable)
            conv_res = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding='SAME')
            res = tf.nn.bias_add(conv_res, biases)
            res = batch_norm(res, True)
            out = tf.nn.relu(res, name=name)

        self.parameters += [kernel, biases]
        return out

    def fc(self, name, input_data, out_channel, is_output=False):
        shape = input_data.get_shape().as_list()
        if len(shape) == 4:
            size = shape[-1] * shape[-2] * shape[-3]
        else:
            size = shape[1]
        input_data_flat = tf.reshape(input_data, [-1, size])
        with tf.variable_scope(name):
            weights = tf.get_variable(name="weights", shape=[size, out_channel], dtype=tf.float32)
            biases = tf.get_variable(name="biases", shape=[out_channel], dtype=tf.float32)
            res = tf.matmul(input_data_flat, weights)
            out = tf.nn.bias_add(res, biases)
            if is_output is False:
                out = batch_norm(out, True, False)
                out = tf.nn.relu(out, name=name)

        self.parameters += [weights, biases]
        return out

    def convlayers(self):
        # zero-mean input
        # conv1
        self.conv1_1 = self.conv("conv1_1", self.imgs, 64)
        self.conv1_2 = self.conv("conv1_2", self.conv1_1, 64)
        self.pool1 = self.maxpool("pool1", self.conv1_2)

        # conv2
        self.conv2_1 = self.conv("conv2_1", self.pool1, 128)
        self.conv2_2 = self.conv("conv2_2", self.conv2_1, 128)
        self.pool2 = self.maxpool("pool2", self.conv2_2)

        # conv3
        self.conv3_1 = self.conv("conv3_1", self.pool2, 256)
        self.conv3_2 = self.conv("conv3_2", self.conv3_1, 256)
        self.conv3_3 = self.conv("conv3_3", self.conv3_2, 256)
        self.pool3 = self.maxpool("pool3", self.conv3_3)

        # conv4
        self.conv4_1 = self.conv("conv4_1", self.pool3, 512)
        self.conv4_2 = self.conv("conv4_2", self.conv4_1, 512)
        self.conv4_3 = self.conv("conv4_3", self.conv4_2, 512)
        self.pool4 = self.maxpool("pool4", self.conv4_3)

        # conv5
        self.conv5_1 = self.conv("conv5_1", self.pool4, 512)
        self.conv5_2 = self.conv("conv5_2", self.conv5_1, 512)
        self.conv5_3 = self.conv("conv5_3", self.conv5_2, 512)
        self.pool5 = self.maxpool("pool5", self.conv5_3)

    def fc_layers(self):
        self.fc6 = self.fc("fc1", self.pool5, 4096)
        # self.fc6 = tf.nn.dropout(self.fc6, 0.5)

        self.fc7 = self.fc("fc2", self.fc6, 4096)
        # self.fc7 = tf.nn.dropout(self.fc7, 0.5)

        self.fc8 = self.fc("fc3", self.fc7, num_classes, is_output=True)

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            sess.run(self.parameters[i].assign(weights[k]))
        print("-----------all done---------------")


if __name__ == '__main__':
    X = tf.placeholder(tf.float32, [None, input_image_shape[0], input_image_shape[1], input_image_shape[2]])
    y = tf.placeholder(tf.float32, [None, num_classes])
    learning_rate_holder = tf.placeholder(tf.float32)

    vgg = vgg16(X)
    prob = vgg.probs

    with tf.name_scope("cross_ent"):
        y_output = tf.nn.softmax(prob)
        cross_entropy = -tf.reduce_sum(y * tf.log(y_output))
        loss = tf.reduce_mean(cross_entropy)

    # Train op
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_holder)
        train_op = optimizer.minimize(loss)

    # Evaluation op: Accuracy of the model
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(y_output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    # init = tf.glorot_normal_initializer()  # failed, 也称之为 Xavier normal initializer. 参考文献[A]

    loss_buf = []
    accuracy_buf = []
    with tf.Session() as sess:
        sess.run(init)

        # Load the pretrained weights into the non-trainable layer
        # model.load_initial_weights(sess)

        total_batch = mnist_data_set.train.num_examples // batch_size
        for step in range(training_epoch):
            print("{} Epoch number: {}".format(datetime.now(), step + 1))

            tmp_loss = []
            for iteration in range(total_batch):
                batch_xs, batch_ys = mnist_data_set.train.next_batch(batch_size)
                batch_xs = image_shape_scale(batch_xs, input_image_shape)

                if step < 10:
                    sess.run(train_op, feed_dict={X: batch_xs, y: batch_ys, learning_rate_holder: learning_rate})
                elif step < 20:
                    sess.run(train_op, feed_dict={X: batch_xs, y: batch_ys, learning_rate_holder: learning_rate/10.0})
                elif step < 30:
                    sess.run(train_op, feed_dict={X: batch_xs, y: batch_ys, learning_rate_holder: learning_rate/100.0})
                else:
                    sess.run(train_op, feed_dict={X: batch_xs, y: batch_ys, learning_rate_holder: learning_rate/1000.0})

                if iteration % 50 == 0:
                    loss_val = sess.run(loss, feed_dict={X: batch_xs, y: batch_ys})
                    train_accuracy = sess.run(accuracy, feed_dict={X: batch_xs, y: batch_ys})
                    print("step {}, iteration {}, loss {}, training accuracy {}".format(step, iteration, loss_val, train_accuracy))

            _loss_buf = []
            _accuracy_buf = []
            test_total_batch = mnist_data_set.test.num_examples // batch_size
            for iteration in range(test_total_batch):
                batch_xs, batch_ys = mnist_data_set.test.next_batch(batch_size)  # GPU内存不足，只好分批测试准确率
                batch_xs = image_shape_scale(batch_xs, input_image_shape)

                loss_val = sess.run(loss, feed_dict={X: batch_xs, y: batch_ys})
                test_accuracy = sess.run(accuracy, feed_dict={X: batch_xs, y: batch_ys})

                _loss_buf.append(loss_val)
                _accuracy_buf.append(test_accuracy)
            loss_val = np.array(_loss_buf).mean()
            test_accuracy = np.array(_accuracy_buf).mean()
            print("step {}, loss {}, testing accuracy {}".format(step, loss_val, test_accuracy))
            loss_buf.append(loss_val)
            accuracy_buf.append(test_accuracy)

# 画出准确率曲线
accuracy_ndarray = np.array(accuracy_buf)
accuracy_size = np.arange(len(accuracy_ndarray))
plt.plot(accuracy_size, accuracy_ndarray, 'b+', label='accuracy')

loss_ndarray = np.array(loss_buf)
loss_size = np.arange(len(loss_ndarray))
plt.plot(loss_size, loss_ndarray, 'r*', label='loss')

plt.show()

# 保存loss和测试准确率到csv文件
with open('VGGNet16-batchnorm.csv', 'w') as fid:
    for loss, acc in zip(loss_buf, accuracy_buf):
        strText = str(loss) + ',' + str(acc) + '\n'
        fid.write(strText)
fid.close()

print('end')

# 参考文献
# [A]：tensorflow参数初始化, https://blog.csdn.net/m0_37167788/article/details/79073070


'''
step 1, loss 2.2608318, testing accuracy 0.9890825
step 2, loss 1.5252224, testing accuracy 0.9931891
... ...
step 49, loss 1.1548673, testing accuracy 0.99348956
step 50, loss 1.1780967, testing accuracy 0.9938902
'''


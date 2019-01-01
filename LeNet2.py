import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

datapath = '/home/xiajun/res/MNIST_data'
mnist_data_set = input_data.read_data_sets(datapath, validation_size=0, one_hot=True)

with tf.device('/cpu:0'):
    learning_rate = 1e-4
    batch_size = 200
    n_features = 28*28*1
    n_classes = 10
    n_fc1 = 7*7*64
    n_fc2 = 1024

    X = tf.placeholder(tf.float32, [None, n_features])
    y = tf.placeholder(tf.float32, [None, n_classes])

    W_conv = {
        'conv1': tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.0001)),
        'conv2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.01)),
        'fc1': tf.Variable(tf.truncated_normal([n_fc1, n_fc1], stddev=0.1)),
        'fc2': tf.Variable(tf.truncated_normal([n_fc1, n_fc2], stddev=0.1)),
        'fc3': tf.Variable(tf.truncated_normal([n_fc2, n_classes], stddev=0.1))
    }
    b_conv = {
        'conv1': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[32])),
        'conv2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[64])),
        'fc1': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc1])),
        'fc2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc2])),
        'fc3': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[n_classes]))
    }

    X_image = tf.reshape(X, [-1, 28, 28, 1])

    # 卷积层1
    conv1 = tf.nn.conv2d(X_image, W_conv['conv1'], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, b_conv['conv1'])
    conv1 = tf.nn.relu(conv1)

    # 池化层1
    pool1 = tf.nn.avg_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # LRN层
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

    # 卷积层2
    conv2 = tf.nn.conv2d(norm1, W_conv['conv2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, b_conv['conv2'])
    conv2 = tf.nn.relu(conv2)
    # LRN层
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

    # 池化层2
    pool2 = tf.nn.avg_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    reshape = tf.reshape(pool2, [-1, n_fc1])

    fc1 = tf.add(tf.matmul(reshape, W_conv['fc1']), b_conv['fc1'])
    fc1 = tf.nn.relu(fc1)

    # 全连接层2
    fc2 = tf.add(tf.matmul(fc1, W_conv['fc2']), b_conv['fc2'])
    fc2 = tf.nn.relu(fc2)

    # 全连接层3
    fc3 = tf.add(tf.matmul(fc2, W_conv['fc3']), b_conv['fc3'])

    y_output = tf.nn.softmax(fc3)

    # 损失函数
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc3, y))  # failed
    cross_entropy = -tf.reduce_sum(y * tf.log(y_output))
    # 使用GD优化算法来调整参数
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # 评估模型
    correct_pred = tf.equal(tf.argmax(fc3, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()


''
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
''


with tf.Session() as sess:
    sess.run(init)
    c = []

    total_batch = mnist_data_set.train.num_examples//batch_size
    for i in range(200):
        for iteration in range(total_batch):
            batch_xs, batch_ys = mnist_data_set.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={X: batch_xs, y: batch_ys})

        batch_xs, batch_ys = mnist_data_set.test.images, mnist_data_set.test.labels
        # train_accuracy = accuracy.eval(feed_dict={X: batch_xs, y: batch_ys})
        train_accuracy = sess.run(accuracy, feed_dict={X: batch_xs, y: batch_ys})
        print("step %d, training accuracy %g" % (i, train_accuracy))

"""
# 程序20-1：下载flowers数据集 （代码下载失败，我是手动下载的）
import tensorflow as tf
from datasets import dataset_utils	#这里请读者一定先下载相应的数据集处理文件夹并先导入到工程中

url = "http://download.tensorflow.org/data/flowers.tar.gz"
flowers_data_dir = './flowers/'
if not tf.gfile.Exists(flowers_data_dir):
    tf.gfile.MakeDirs(flowers_data_dir)

dataset_utils.download_and_uncompress_tarball(url, flowers_data_dir)
"""


"""
# 程序20-2：下载flowers数据集是tfrecord格式的，本代码是动过slim读取tfrecord数据。
from datasets import flowers
import tensorflow as tf
import matplotlib.pyplot as plt
# import global_variable
import tensorflow.contrib.slim.python.slim as slim
flowers_data_dir = './flowers/'

with tf.Graph().as_default():
    dataset = flowers.get_split('train', flowers_data_dir)
    import tensorflow.contrib.slim.python.slim.data.dataset_data_provider as providerr
    data_provider = providerr.DatasetDataProvider(
        dataset, common_queue_capacity=32, common_queue_min=1)
    image, label = data_provider.get(['image', 'label'])

    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            for i in range(5):
                np_image, np_label = sess.run([image, label])
                height, width, _ = np_image.shape
                class_name = name = dataset.labels_to_names[np_label]

                plt.figure()
                plt.imshow(np_image)
                plt.title('%s, %d x %d' % (name, height, width))
                plt.axis('off')
                plt.show()
"""


"""
# 程序20-4
import tensorflow as tf
import slim_cnn_model as model

with tf.Graph().as_default():
    image = tf.random_normal([1,217,217,3])
    probabilities = model.Slim_cnn(image,5)
    probabilities = tf.nn.softmax(probabilities.net)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(probabilities)

        print('Res Shape:')
        print(res.shape)

        print('\nRes:')
        print(res)
"""

import tensorflow as tf
import slim_cnn_model as model
from datasets import flowers
import tensorflow.contrib.slim as slim

flowers_data_dir = './flowers/'
save_model = './save_model'


def load_batch(dataset, batch_size=32, height=217, width=217, is_training=True):
    import tensorflow.contrib.slim.python.slim.data.dataset_data_provider as providerr
    data_provider = providerr.DatasetDataProvider(
        dataset, common_queue_capacity=32, common_queue_min=1)
    image_raw, label = data_provider.get(['image', 'label'])
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.image.convert_image_dtype(image_raw, tf.float32)

    images_raw, labels = tf.train.batch(
        [image_raw, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=2 * batch_size)
    return images_raw, labels


with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)

    dataset = flowers.get_split('train', flowers_data_dir)
    images, labels = load_batch(dataset)

    probabilities = model.Slim_cnn(images, 5)
    probabilities = tf.nn.softmax(probabilities.net)

    one_hot_labels = slim.one_hot_encoding(labels, 5)
    slim.losses.softmax_cross_entropy(probabilities, one_hot_labels)
    total_loss = slim.losses.get_total_loss()

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    final_loss = slim.learning.train(
        train_op,
        logdir=save_model,
        number_of_steps=100
    )

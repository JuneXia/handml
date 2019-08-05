import os
import argparse
import tensorflow as tf
import datasets
import network
import losses
import metrics


class MNistNet(object):
    def __init__(self, n_classes, epoch_size):
        super(MNistNet, self).__init__()
        self.epoch_size = epoch_size

        self.global_step = tf.Variable(tf.constant(0), trainable=False)
        self.init_learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, ), name='labels')
        self.images_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input')

        embedding_net = network.LeNet
        # siamese_net = SiameseNet(embedding_net)
        classification_net = network.ClassificationNet(self.images_placeholder, embedding_net, n_classes=n_classes)

        # TODO: 损失函数可以抽象重构一下。
        # regularization_loss = losses.RegularizationLoss(classification_net.prelogits)
        classification_loss = losses.ClassificationLoss(self.labels_placeholder, classification_net.logits)
        self.total_loss = losses.TotalLoss().total_loss
        self.accuracy = metrics.AccMetric(self.labels_placeholder, classification_net.logits).accuracy

        learning_rate_decay_epochs = 50
        learning_rate_decay_factor = 0.9
        self.learning_rate = tf.train.exponential_decay(self.init_learning_rate_placeholder, self.global_step,
                                                        learning_rate_decay_epochs * self.epoch_size,
                                                        learning_rate_decay_factor, staircase=True)

        """
        moving_average_decay = 0.9999
        self.train_op = facenet.train(self.total_loss, self.global_step, "ADAM",
                                 self.learning_rate, moving_average_decay, tf.global_variables(),
                                 log_histograms=False)
        """

        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.total_loss)

    def train_start(self, dataset, max_nrof_epochs=500, validate_every_n_epochs=5, model_save_path='./save_model'):
        gpu_memory_fraction = 0.7
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            global_step = 0
            try:
                count = 0
                while True:
                    global_step = sess.run(self.global_step, feed_dict={self.global_step: count})
                    images, labels = sess.run(dataset.get_next)
                    feed_dict = {self.init_learning_rate_placeholder: 0.01, self.labels_placeholder: labels, self.images_placeholder: images}
                    tensor_list = [self.total_loss, self.train_op, self.accuracy, self.learning_rate]

                    total_loss, _, acc, learning_rate = sess.run(tensor_list, feed_dict=feed_dict)
                    print('step %d\ttotal_loss %.3f\tacc %.3f\tlearning_rate %.5f' % (global_step, total_loss, acc, learning_rate))

                    if global_step / self.epoch_size > 10000:
                        checkpoint_name = os.path.join(model_save_path, 'model-%s.ckpt' % str(global_step / self.epoch_size))
                        saver.save(sess, checkpoint_name)

                    count += 1
            except tf.errors.OutOfRangeError:
                print('end!')

            checkpoint_name = os.path.join(model_save_path, 'model-%s.ckpt' % str(global_step / self.epoch_size))
            saver.save(sess, checkpoint_name)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('embeddings_file', type=str,
                        help='Path to the embeddings file.')
    parser.add_argument('--model_savepath', type=str,
                        help='model save path.', default='')
    parser.add_argument('--test_model_path', type=str,
                        help='test model path.', default='')
    return parser.parse_args(argv)


if __name__ == '__main__':
    data_path = '/home/xiajun/res/MNIST_data/train'

    batch_size = 128
    repeat = 50
    buffer_size = 10000
    mnist_dataset = datasets.DataSet(data_path, batch_size=batch_size, repeat=repeat, buffer_size=buffer_size)
    # siamese_dataset = SiameseMNIST(VGGFaceDataset)

    mnist_net = MNistNet(mnist_dataset.n_classes, mnist_dataset.epoch_size)
    mnist_net.train_start(mnist_dataset)
    print('debug')




"""
TODO: 先搞定“加载训练好的模型，可视化embedding”，因为facenet的模型已经训练好了。
然后搞定训练时动态显示embedding，方便后期训练观察。

TODO: datasets.DataSet中得到的既有训练集也有测试集，但是目前只用到了训练集。
1. 先得到训练集和测试集，然后分别生成两个类，就像PyTorch实现的siamese-triplet一样；
2. 直接得到一个类，该类中同时包含训练集和测试集，需要用什么自己去取。
"""





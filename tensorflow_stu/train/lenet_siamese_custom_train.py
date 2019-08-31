import os
import sys
project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_path)

from datasets import dataset as datset
import tensorflow as tf
from utils import util

import socket
import getpass


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
home_path = os.environ['HOME']

user_name = getpass.getuser()
host_name = socket.gethostname()

if user_name == 'xiajun':
    g_datapath = os.path.join(home_path, 'res/mnist/train')
elif user_name == 'xiaj':
    g_datapath = os.path.join(home_path, 'res/mnist')
else:
    print('unkown user_name:{}'.format(user_name))
    exit(0)

# Change Mark: 相对于lenet2.py，本次实验必须使用eager模式，否则会报错
tf.enable_eager_execution()
print('is eager executing: ', tf.executing_eagerly())


class ContrastiveLoss(tf.keras.layers.Layer):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def __call__(self, *args, **kwargs):
        (pred1, pred2), labels = args

        eps = 1e-9
        margin = 1.
        size_average = True
        square = tf.pow(pred2 - pred1, 2)
        distances = tf.reduce_sum(square, axis=1)

        tmp2 = tf.keras.activations.relu(margin - tf.sqrt(distances + eps))
        tmp2 = tf.pow(tmp2, 2)
        tmp1 = labels * distances + (1 + -1 * labels) * tmp2
        losses = 0.5 * tmp1
        if size_average:
            loss_step = tf.reduce_mean(losses)
        else:
            loss_step = tf.reduce_sum(losses)

        return loss_step


class SiameseNet(tf.keras.layers.Layer):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def call(self, inputs, **kwargs):
        output1 = self.embedding_net(inputs[0])
        output2 = self.embedding_net(inputs[1])

        return output1, output2


class ClassificationNet(tf.keras.layers.Layer):
    def __init__(self, embedding_net, num_class):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        # self.nonlinear = tf.keras.activations.relu()  # not for custom train
        # self.nonlinear = tf.keras.layers.PReLU()
        # self.nonlinear = tf.keras.layers.Activation()
        self.nonlinear = tf.keras.layers.ReLU()
        self.fc1 = tf.keras.layers.Dense(num_class, activation='softmax')

    def call(self, inputs, **kwargs):
        output = self.embedding_net(inputs)
        output = self.nonlinear(output)
        scores = self.fc1(output)

        return scores


class AccumulatedLossMetric(tf.keras.layers.Layer):
    def __init__(self, name='Loss'):
        super(AccumulatedLossMetric, self).__init__()
        self.metric_loss = tf.keras.metrics.Mean(name)

    def __call__(self, *args, **kwargs):
        predicts, labels, loss_step = args
        return self.metric_loss(loss_step)

    def result(self):
        return self.metric_loss.result()

    def reset_states(self):
        self.metric_loss.reset_states()


class OneShotNet(tf.keras.layers.Layer):
    def __init__(self, embedding_net):
        super(OneShotNet, self).__init__()
        self.embedding_net = embedding_net
        self.L1 = tf.keras.regularizers.l1()

    def call(self, inputs, **kwargs):
        >>>
        l1_dist = tf.abs(inputs[0]-inputs[1])

        print('debug')




class Train(object):
    def __init__(self, model, loss_func, dataset, optimizer, metrics=[]):
        super(Train, self).__init__()

        self.model = model
        self.loss_func = loss_func
        self.dataset = dataset
        self.optimizer = optimizer
        self.metrics = metrics

    def start(self):
        for epoch in range(10):
            for batch, (images, labels) in enumerate(dataset):
                with tf.GradientTape() as t:
                    outputs = self.model(images)
                    loss_step = self.loss_func(outputs, labels)

                grads = t.gradient(loss_step, model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

                for metric in self.metrics:
                    metric(outputs, labels, loss_step)

                    print('{}'.format(metric.result()))


if __name__ == '__main__':
    images_path, images_label = util.get_dataset(g_datapath)
    num_class = len(set(images_label))
    batch_size = 100
    dataset = datset.SiameseDataset(images_path, images_label)
    dataset = datset.DataIterator(dataset, batch_size=batch_size)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, [3, 3], activation='relu', input_shape=(None, None, 1)),
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.GlobalMaxPool2D(),
        tf.keras.layers.Dense(10)
    ])
    # model = ClassificationNet(model, num_class)
    model = SiameseNet(model)
    # model = OneShotNet(model)
    model = tf.keras.Sequential([model, OneShotNet(model)])

    # optimizer = tf.keras.optimizers.Adam()
    optimizer = tf.train.AdamOptimizer()

    # loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss_func = ContrastiveLoss()

    # metric_train_loss = tf.keras.metrics.Mean('train_loss')
    # metric_train_acc = tf.keras.metrics.CategoricalAccuracy('train_acc')
    # metric_test_loss = tf.keras.metrics.Mean('test_loss')
    # metric_test_acc = tf.keras.metrics.CategoricalAccuracy('test_acc')
    metrics = [AccumulatedLossMetric('train_loss')]

    trainer = Train(model, loss_func, dataset, optimizer, metrics=metrics)
    trainer.start()

    print('debug')



"""
reference:
https://zhuanlan.zhihu.com/p/66648325
https://tf.wiki/zh/basic/models.html
https://www.zybuluo.com/Team/note/1491361
"""
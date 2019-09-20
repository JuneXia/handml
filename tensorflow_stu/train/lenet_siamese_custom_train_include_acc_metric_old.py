# -*- coding: UTF-8 -*-
# /usr/
import os
import sys
project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_path)

from datasets import dataset as datset
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from utils import util
import datetime
import numpy as np

import socket
import getpass


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
home_path = os.environ['HOME']

user_name = getpass.getuser()
host_name = socket.gethostname()

if user_name in ['xiajun', 'yp']:
    g_datapath = os.path.join(home_path, 'res/mnist/train')
elif user_name == 'xiaj':
    g_datapath = os.path.join(home_path, 'res/mnist')
else:
    print('unkown user_name:{}'.format(user_name))
    exit(0)

# Change Mark: 相对于lenet2.py，本次实验必须使用eager模式，否则会报错
tf.enable_eager_execution()
print('is eager executing: ', tf.executing_eagerly())

import numpy.random as rng

def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    # return K.variable(values,name=name)
    return tf.Variable(values,name=name)
#//TODO: figure out how to initialize layer biases in keras.
def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    # return K.variable(values,name=name)
    return tf.Variable(values,name=name)


class ContrastiveLoss(tf.keras.layers.Layer):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def __call__(self, *args, **kwargs):
        labels, (pred1, pred2) = args

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


class ComplexLoss(tf.keras.layers.Layer):
    def __init__(self):
        super(ComplexLoss, self).__init__()
        self.bincross_loss_func = tf.keras.losses.BinaryCrossentropy()
        self.contrast_loss_func = ContrastiveLoss()

    def __call__(self, *args, **kwargs):
        labels, outputs, (emb1, emb2) = args
        bincross_loss = self.bincross_loss_func(labels, outputs)
        contrast_loss = self.contrast_loss_func(labels, (emb1, emb2))

        # ok
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.add_loss(bincross_loss, inputs=True)
        self.add_loss(contrast_loss, inputs=True)
        '''
        Epoch 9 batch 620 train loss:0.25426143407821655, train acc:0.8569740653038025
Epoch 9 batch 640 train loss:0.25419411063194275, train acc:0.856954038143158
Epoch 9 batch 660 train loss:0.25412696599960327, train acc:0.8571060299873352
Epoch 9 batch 680 train loss:0.2540600299835205, train acc:0.8571428656578064
        '''
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # loss = tf.reduce_mean((bincross_loss, contrast_loss))
        # self.add_loss(loss, inputs=True)
        '''
        Epoch 9 batch 620 train loss:0.2688220739364624, train acc:0.8453602194786072
Epoch 9 batch 640 train loss:0.2687385082244873, train acc:0.8454023003578186
Epoch 9 batch 660 train loss:0.268655389547348, train acc:0.845386803150177
Epoch 9 batch 680 train loss:0.2685726284980774, train acc:0.8453999757766724
        '''

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        return self.losses

    def call_need_tuple_when_invoke_function(self, inputs, **kwargs):  # 调用函数时需要传元祖
        labels, outputs, (emb1, emb2) = inputs
        bincross_loss = self.bincross_loss_func(labels, outputs)
        contrast_loss = self.contrast_loss_func(labels, (emb1, emb2))
        self.add_loss(bincross_loss, inputs=True)
        self.add_loss(contrast_loss, inputs=True)

        # self.add_metric(bincross_loss)


class SiameseNet(tf.keras.Model):
    """
    There are not save_weights function if inherit tf.keras.layers.Layer
    """
    def __init__(self, embedding_net, imshape=(28, 28, 1)):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.l1_layer = tf.keras.layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
        self.fc1 = tf.keras.layers.Dense(1, activation='sigmoid')  # TODO: This fc1_layer should not in the siamese-network.
        self.predicts = None

        # TODO: Just to get model variable explicit
        data = np.ones((10, ) + imshape, dtype=np.float32)
        self.call((data, data))

    def build_failed(self, input_shape):
        '''
        for batch, (images, labels) in enumerate(self.validation_dataset):
            outputs = self.model(images)
            labels = labels.reshape((-1, 1))
            loss_step = self.loss_func(labels, outputs, (self.model.emb1, self.model.emb2))
        '''

        shape = tf.TensorShape((input_shape[1], self.output_dim))
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=True)
        # Be sure to call this at the end
        super(SiameseNet, self).build(input_shape)


    '''
    def build(self, input_shape):
        # The build method gets called the first time your layer is used.
        # Creating variables on build() allows you to make their shape depend
        # on the input shape and hence removes the need for the user to specify
        # full shapes. It is possible to create variables during __init__() if
        # you already know their full shapes.
        self.kernel = self.add_variable(
            "kernel", [input_shape[-1], self.output_units])
    '''

    def call(self, inputs, **kwargs):
        emb1 = self.embedding_net(inputs[0])
        emb2 = self.embedding_net(inputs[1])
        self.emb1 = emb1
        self.emb2 = emb2

        l1_dist = self.l1_layer([emb1, emb2])

        # l1_dist = tf.reduce_sum(tf.abs(emb1 - emb2), axis=1)
        # l1_dist = tf.expand_dims(l1_dist, axis=1)

        predicts = self.fc1(l1_dist)

        return predicts

    def get_output(self):
        return self.predicts


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
        labels, predicts, loss_step = args
        return self.metric_loss(loss_step)

    def name(self):
        return self.metric_loss.name

    def result(self):
        return self.metric_loss.result()

    def reset_states(self):
        self.metric_loss.reset_states()


class AccumulatedAccuaracyMetric(tf.keras.layers.Layer):
    def __init__(self, name='acc'):
        super(AccumulatedAccuaracyMetric, self).__init__()
        self.metric_acc = tf.keras.metrics.Accuracy(name)

    def __call__(self, *args, **kwargs):
        labels, predicts, loss_step = args
        return self.metric_acc(labels, predicts)

    def name(self):
        return self.metric_acc.name

    def result(self):
        return self.metric_acc.result()

    def reset_states(self):
        self.metric_acc.reset_states()


class AccumulatedBinaryAccuracyMetric(tf.keras.layers.Layer):
    def __init1__(self, name='acc'):
        super(AccumulatedBinaryAccuracyMetric, self).__init__()
        self.metric_acc = tf.keras.metrics.BinaryAccuracy(name)

    def __init__(self, metric_fuc):
        super(AccumulatedBinaryAccuracyMetric, self).__init__()
        self.metric_acc = metric_fuc

    def __call__(self, *args, **kwargs):
        labels, predicts, loss_step = args
        return self.metric_acc(labels, predicts)

    def name(self):
        return self.metric_acc.name

    def result(self):
        return self.metric_acc.result()

    def reset_states(self):
        self.metric_acc.reset_states()


class OneShotNet(tf.keras.layers.Layer):
    def __init__(self, embedding_net):
        super(OneShotNet, self).__init__()
        self.embedding_net = embedding_net
        self.L1 = tf.keras.regularizers.l1()

    def call(self, inputs, **kwargs):
        l1_dist = tf.abs(inputs[0]-inputs[1])

        print('debug')


class Checkpint(tf.keras.callbacks.Callback):
    def __init__(self, save_path):
        super(Checkpint, self).__init__()
        self.save_path = save_path
        # tf.keras.callbacks.ModelCheckpoint(filepath)

        self.checkpoint_path = os.path.join(self.save_path, 'ckpt')

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

        # tensorflow 1.x适用
        self.saver = tfe.Saver(self.model.variables)

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        # TODO: 这里使用epoch代替 global_step，因为self.saver.save函数没有epoch参数。
        self.saver.save(self.checkpoint_path, global_step=epoch)


def SaveCheckpoint(model, save_path, global_step):
    checkpoint_path = os.path.join(save_path, 'ckpt')

    # tensorflow 1.x适用
    tfe.Saver(model.variables).save(checkpoint_path, global_step=global_step)

    # 发现模型恢复时，不能获取动态图的变量，可能需要在模型__init__中add_variable或者add_weights

    # model.save_weights(checkpoint_path)  # ok, 这个应该是只能保存最后一次weights
    # model.save(os.path.join('./save_model', 'mymodel.h5'))  # failed

    # tensorflow 2.0适用
    # checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model, optimizer_step=self.global_step)
    # checkpoint.save(file_prefix=checkpoint_path)

def RestoreCheckpoint(model, model_path):
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint(model_path))

    print('debug')
    # model.load_weights(checkpoint_path)

    # saver = tfe.Saver(model.variables)
    # saver.restore(tf.train.latest_checkpoint(checkpoint_path))


class Train(object):
    def __init__(self, model, loss_func, train_dataset=None, validation_dataset=None, optimizer=None, metrics=[]):
        super(Train, self).__init__()

        self.model = model
        self.loss_func = loss_func
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.optimizer = optimizer
        self.metrics = metrics

        if train_dataset is not None:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = os.path.join(project_path, 'logs', current_time)
            # summary_writer 一定要写成成员变量，因为这会常住内存，否则会报错。
            self.summary_writer = tf.contrib.summary.create_file_writer(train_log_dir, flush_millis=1000)
            self.global_step = tf.train.get_or_create_global_step()
            self.summary_writer.set_as_default()

    def start(self, epoch_size=1, checkpoint=None):
        self.epoch_size = epoch_size
        checkpoint.set_model(model)
        with tf.contrib.summary.record_summaries_every_n_global_steps(5, self.global_step):
            for epoch in range(self.epoch_size):
                self.train(epoch)
                self.validation()
                checkpoint.on_epoch_end(epoch)

    def train(self, epoch):
        for metric in self.metrics:
            metric.reset_states()

        for batch, (images, labels) in enumerate(self.train_dataset):
            self.global_step.assign_add(1)
            with tf.GradientTape() as t:
                outputs = self.model(images)
                labels = labels.reshape((-1, 1))
                loss_step = self.loss_func(labels, outputs, (self.model.emb1, self.model.emb2))

            grads = t.gradient(loss_step, model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

            print('Training: global step: {}, epoch: {}/{}, num_batch: {}'.format(self.global_step.numpy(), epoch,
                                                                        self.epoch_size,
                                                                        batch), end=' ')
            for metric in self.metrics:
                rslt = metric(labels, outputs, loss_step)
                tf.contrib.summary.scalar('train_' + metric.name(), rslt)
                print(metric.name(), rslt.numpy(), end=' ')
            print('')

    def validation(self):
        embedding_visualize = True
        if embedding_visualize:
            output_list = []
            label_list = []
        for metric in self.metrics:
            metric.reset_states()

        for batch, (images, labels) in enumerate(self.validation_dataset):
            outputs = self.model(images)
            labels = labels.reshape((-1, 1))
            loss_step = self.loss_func(labels, outputs, (self.model.emb1, self.model.emb2))

            if embedding_visualize:
                output_list.extend(outputs)
                label_list.extend(labels)

            for metric in self.metrics:
                rslt = metric(labels, outputs, loss_step)
                tf.contrib.summary.scalar('validate_' + metric.name(), rslt)

        print('Validate: ', end=' ')
        for metric in self.metrics:
            print(metric.name(), metric.result().numpy(), end=' ')
        print('')

        if embedding_visualize:
            output_list = np.array(output_list)
            # EmbVisual.visualisation(output_list)


def learning_rate_sche(epoch):
    learning_rate = 0.2
    if epoch > 5:
        learning_rate = 0.02
    elif epoch > 10:
        learning_rate = 0.01

    return learning_rate


class DummyFileWriter(object):
  def get_logdir(self):
    return './logs'



def projector_embedding():
    '''
        # self._writer = tf.contrib.summary.create_file_writer('path')
        embedding_config = projector.ProjectorConfig()
        embedding = embedding_config.embeddings.add()
        embedding.tensor_name = emb.name
        embedding.metadata_path = 'metadata.tsv'
        # projector.visualize_embeddings(train_writer, embedding_config)
        projector.visualize_embeddings(DummyFileWriter(), embedding_config)
    '''
    pass


if __name__ == '__main__1':
    train_images_path, train_images_label, validation_images_path, validation_images_label = datset.load_dataset(g_datapath, validation_ratio=0.2)
    num_class = len(set(train_images_label))
    batch_size = 100
    train_dataset = datset.SiameseDataset(train_images_path, train_images_label)
    validation_dataset = datset.SiameseDataset(validation_images_path, validation_images_label, is_train=False)
    train_dataset = datset.DataIterator(train_dataset, batch_size=batch_size)
    validation_dataset = datset.DataIterator(validation_dataset, batch_size=batch_size)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, [3, 3], activation='relu', input_shape=(None, None, 1)),
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.GlobalMaxPool2D(),
        tf.keras.layers.Dense(10)
    ])
    emb = model.output
    # model = ClassificationNet(model, num_class)
    model = SiameseNet(model)
    # model = OneShotNet(model)
    # model = tf.keras.Sequential([model, OneShotNet(model)])

    # optimizer = tf.keras.optimizers.Adam()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

    # loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # contrast_loss_func = ContrastiveLoss()
    # loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss_func = ComplexLoss()

    metric_train_acc = AccumulatedBinaryAccuracyMetric(tf.keras.metrics.BinaryAccuracy('bin_acc'))
    metrics = [AccumulatedLossMetric('loss'), metric_train_acc]

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(learning_rate_sche)
    checkpoint_save_path = './save_model'
    checkpoint = Checkpint(checkpoint_save_path)

    trainer = Train(model, loss_func, train_dataset, validation_dataset, optimizer, metrics=metrics)
    trainer.start(epoch_size=2, checkpoint=checkpoint)

    #  >>> TODO: 看官方文档：https://www.tensorflow.org/guide/keras#weights_only
    ## TODO: 研究下 tf.train.CheckpointManager()
    print('debug')


if __name__ == '__main__':
    train_images_path, train_images_label, validation_images_path, validation_images_label = datset.load_dataset(g_datapath, validation_ratio=0.2)
    num_class = len(set(train_images_label))
    batch_size = 100
    train_dataset = datset.SiameseDataset(train_images_path, train_images_label)
    validation_dataset = datset.SiameseDataset(validation_images_path, validation_images_label, is_train=False)
    train_dataset = datset.DataIterator(train_dataset, batch_size=batch_size)
    validation_dataset = datset.DataIterator(validation_dataset, batch_size=batch_size)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, [3, 3], activation='relu', input_shape=(None, None, 1)),
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.GlobalMaxPool2D(),
        tf.keras.layers.Dense(10)
    ])
    emb = model.output
    # model = SiameseNet(model)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

    loss_func = ComplexLoss()

    metric_train_acc = AccumulatedBinaryAccuracyMetric(tf.keras.metrics.BinaryAccuracy('bin_acc'))
    metrics = [AccumulatedLossMetric('loss'), metric_train_acc]

    model_path = './save_model'
    RestoreCheckpoint(model, model_path)

    trainer = Train(model, loss_func, validation_dataset=validation_dataset, metrics=metrics)
    trainer.validation()

    print('debug')





"""
reference:
https://zhuanlan.zhihu.com/p/66648325
两个embedding的合并参考了这篇文章：https://blog.csdn.net/huowa9077/article/details/81082795
https://tf.wiki/zh/basic/models.html
https://www.zybuluo.com/Team/note/1491361
"""



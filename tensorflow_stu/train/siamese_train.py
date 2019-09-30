# -*- coding: UTF-8 -*-
# /usr/
import os
import sys
project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_path)

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from utils import tools
from utils import dataset as datset
from networks import network
from losses import loss
from metrics import metric
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

        # model.save_weights(checkpoint_path)  # ok, 但是这个应该是只能保存最后一次weights
        # model.save(os.path.join('./save_model', 'mymodel.h5'))  # failed

        # tensorflow 2.0适用
        # checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model, optimizer_step=self.global_step)
        # checkpoint.save(file_prefix=checkpoint_path)

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        # TODO: 这里使用epoch代替 global_step，因为self.saver.save函数没有epoch参数。
        self.saver.save(self.checkpoint_path, global_step=epoch)


def RestoreCheckpoint(model, model_path):
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint(model_path))

    print('debug')
    # model.load_weights(checkpoint_path)

    # saver = tfe.Saver(model.variables)
    # saver.restore(tf.train.latest_checkpoint(checkpoint_path))


class Train(object):
    def __init__(self, model, loss_func, train_dataset=None, validation_dataset=None, metrics=[]):
        super(Train, self).__init__()

        self.model = model
        self.loss_func = loss_func
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.optimizer = None
        self.metrics = metrics

        self.train_metrics = []
        self.validate_metrics = []

        if train_dataset is not None:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = os.path.join(project_path, 'train/logs', current_time)
            # summary_writer 一定要写成成员变量，因为这会常住内存，否则会报错。
            self.summary_writer = tf.contrib.summary.create_file_writer(train_log_dir, flush_millis=1000)
            self.global_step = tf.train.get_or_create_global_step()
            self.summary_writer.set_as_default()

    def set_metrics(self, train_metrics=[], validate_metrics=[]):
        """
        TODO:感觉这种写法并不是很好。
        :param train_metrics:
        :param validate_metrics:
        :return:
        """
        self.train_metrics = train_metrics
        self.validate_metrics = validate_metrics

    def set_optimizer(self):
        learning_rate = 0.01
        # learning_rate = tf.train.exponential_decay(0.01, global_step=self.global_step, decay_steps=2, decay_rate=0.03)
        # optimizer = tf.keras.optimizers.Adam()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    def start(self, epoch_size=1, checkpoint=None):
        self.epoch_size = epoch_size
        checkpoint.set_model(self.model)
        with tf.contrib.summary.record_summaries_every_n_global_steps(5, self.global_step):
            for epoch in range(self.epoch_size):
                self.train(epoch)
                self.validation()
                self.evaluate()
                checkpoint.on_epoch_end(epoch)

    def train(self, epoch):
        for metric in self.metrics:
            metric.reset_states()

        for batch, (images, labels) in enumerate(self.train_dataset):
            self.global_step.assign_add(1)
            with tf.GradientTape() as t:
                outputs = self.model(images)
                labels = labels.reshape((-1, 1))
                loss_step = self.loss_func(labels, outputs)

            grads = t.gradient(loss_step, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            for metric in self.metrics:
                rslt = metric(labels, outputs, loss_step)
                tf.contrib.summary.scalar('train_' + metric.name(), rslt)

            if batch % 10 == 0:
                print('Training: global step: {}, epoch: {}/{}, num_batch: {}'.format(self.global_step.numpy(), epoch,
                                                                                      self.epoch_size,
                                                                                      batch), end=' ')
                for metric in self.metrics:
                    print(metric.name(), metric.result().numpy(), end=' ')
                print('')

    def validation(self):
        for metric in self.metrics:
            metric.reset_states()

        for batch, (images, labels) in enumerate(self.validation_dataset):
            outputs = self.model(images)
            labels = labels.reshape((-1, 1))
            loss_step = self.loss_func(labels, outputs)

            for metric in self.metrics:
                rslt = metric(labels, outputs, loss_step)

        print('Validate: ', end=' ')
        for metric in self.metrics:
            print(metric.name(), metric.result().numpy(), end=' ')
            tf.contrib.summary.scalar('validate_' + metric.name(), rslt)
        print('')

    def evaluate(self):
        if len(self.validate_metrics) == 0:
            return

        for metric in self.validate_metrics:
            metric.reset_states()

        embeddings1 = []
        embeddings2 = []
        emb_labels = []
        losses = []
        for batch, (images, labels) in enumerate(self.validation_dataset):
            predicts, (emb1, emb2) = self.model(images)
            embs1, embs2 = emb1.numpy(), emb2.numpy()
            labels = labels.reshape((-1, 1))
            embeddings1.extend(embs1)
            embeddings2.extend(embs2)
            emb_labels.extend(labels)
            #loss_step = self.loss_func(labels, outputs)
            #losses.append(loss_step)
        embeddings1 = np.array(embeddings1)
        embeddings2 = np.array(embeddings2)
        emb_labels = np.array(emb_labels).flatten()

        for metric in self.validate_metrics:
            rslt = metric(emb_labels, (embeddings1, embeddings2), losses)
            acc, acc_std, auc, val, val_std, far = rslt
            tf.contrib.summary.scalar('evaluate_acc', acc)
            tf.contrib.summary.scalar('evaluate_accstd', acc_std)
            tf.contrib.summary.scalar('evaluate_auc', auc)
            tf.contrib.summary.scalar('evaluate_val', val)
            tf.contrib.summary.scalar('evaluate_valstd', val_std)
            tf.contrib.summary.scalar('evaluate_far', far)

            val = '%.4f±%.3f' % (val, val_std)
            acc = '%.4f±%.3f' % (acc, acc_std)
            auc = '%.4f' % auc
            far = '%.4f' % far

            print('Evaluate: acc:{}, auc={}, val:{}@far:{}'.format(acc, auc, val, far), end=' ')
        print('')


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


def create_model(just_embedding=False):
    # create embedding_net
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, [3, 3], activation='relu', input_shape=(None, None, 1)),
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.GlobalMaxPool2D(),
        tf.keras.layers.Dense(32)
    ])

    if not just_embedding:
        # 思路1：ok
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        model = tf.keras.Sequential([
            network.SiameseNet(model),
            network.ContrastiveBinClassifyNet()
        ])
        data = np.ones((10,) + (28, 28, 1), dtype=np.float32)
        model((data, data))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # 思路2：ok
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # model = SiameseBinClassifyNet(model)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    return model


if __name__ == '__main__':  # train, validation, evaluate
    train_images_path, train_images_label, validation_images_path, validation_images_label = datset.load_dataset(g_datapath, validation_ratio=0.2)
    num_class = len(set(train_images_label))
    batch_size = 100
    train_dataset = datset.SiameseDataset(train_images_path, train_images_label)
    validation_dataset = datset.SiameseDataset(validation_images_path, validation_images_label, is_train=False)
    train_dataset = datset.DataIterator(train_dataset, batch_size=batch_size)
    validation_dataset = datset.DataIterator(validation_dataset, batch_size=batch_size)

    model = create_model()

    loss_func = loss.ComplexLoss()

    loss_metric = metric.AccumulatedLossMetric('loss')
    train_acc_metric = metric.AccumulatedBinaryAccuracyMetric('acc')
    evaluate_acc_metric = metric.AccumulatedEmbeddingAccuracyMetric('evaluate')

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
    # lr_callback = tf.keras.callbacks.LearningRateScheduler(learning_rate_sche)
    checkpoint_save_path = './save_model'

    trainer = Train(model, loss_func, train_dataset, validation_dataset, metrics=[loss_metric, train_acc_metric])
    trainer.set_metrics(validate_metrics=[evaluate_acc_metric])
    if True:
        checkpoint = Checkpint(checkpoint_save_path)
        trainer.set_optimizer()
        trainer.start(epoch_size=500, checkpoint=checkpoint)
    elif False:
        RestoreCheckpoint(model, checkpoint_save_path)

        trainer.validation()
        trainer.evaluate()

    #  >>> TODO: 看官方文档：https://www.tensorflow.org/guide/keras#weights_only
    ## TODO: 研究下 tf.train.CheckpointManager()
    print('debug')


if __name__ == '__main__1':  # plot embedding
    validation_images_path, validation_images_label = datset.load_dataset(g_datapath, max_nrof_cls=100)

    batch_size = 100
    buffer_size = 1000
    repeat = 1

    filenames = tf.constant(validation_images_path)
    labels = tf.constant(validation_images_label)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(datset._parse_function)
    dataset = dataset.shuffle(buffer_size=buffer_size, seed=tf.set_random_seed(666),
                              reshuffle_each_iteration=True).batch(batch_size).repeat(1)
    model = create_model()

    model_path = './save_model'
    RestoreCheckpoint(model, model_path)

    embeddings = []
    emb_labels = []
    for (batch, (images, labels)) in enumerate(dataset):
        print(batch, images.shape, labels.shape)
        embs = model(images)
        embeddings.extend(embs.numpy())
        emb_labels.extend(labels.numpy())
    embeddings = np.array(embeddings)
    emb_labels = np.array(emb_labels)

    result = tools.dim_reduct(embeddings)

    print('plot_embedding')
    tools.plot_embedding(data=result, label=emb_labels, title='t-SNE embedding')

"""
reference:
https://zhuanlan.zhihu.com/p/66648325
两个embedding的合并参考了这篇文章：https://blog.csdn.net/huowa9077/article/details/81082795
https://tf.wiki/zh/basic/models.html
https://www.zybuluo.com/Team/note/1491361
"""

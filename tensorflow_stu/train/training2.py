# -*- coding: UTF-8 -*-
import os
import sys
project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
project2_path = os.path.dirname(project_path)
print('[training]:: project_path: {}'.format(project_path))
print('[training]:: project2_path: {}'.format(project2_path))
sys.path.append(project_path)
sys.path.append(project2_path)

import tensorflow as tf
from tensorflow.python.keras import losses
from utils import dataset as datset
from utils import tools
import numpy as np
from six import iteritems

import socket
import getpass


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
home_path = os.environ['HOME']

user_name = getpass.getuser()
host_name = socket.gethostname()

if user_name in ['xiajun']:
    g_datapath = os.path.join(home_path, 'res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44')
elif user_name in ['yp']:
    g_datapath = os.path.join(home_path, 'res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44_front999')
elif user_name in ['xiaj'] and host_name in ['ubuntu']:
    g_datapath = os.path.join(home_path, 'res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44')
elif user_name in ['xiaj'] and host_name in ['ubuntu-pc']:
    g_datapath = os.path.join(home_path, 'res/mnist')
else:
    print('unkown user_name:{}'.format(user_name))
    exit(0)

tf.enable_eager_execution()  # TODO： 不知何故，目前代码必须要使用eager模式，不然训练过程中会出错。
print('is eager executing: ', tf.executing_eagerly())

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)  # local 0.333
config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


class Config(object):
    def __init__(self):
        pass


class PrelogitsNormLoss(losses.Loss):
    def __init__(self, reduction="sum_over_batch_size", name=None):
        super(PrelogitsNormLoss, self).__init__(reduction=reduction, name=name)

        self.eps = 1e-4
        self.prelogits_norm_p = 1
        self.prelogits_norm_loss_factor = 5e-4
        # prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(model.output) + eps, ord=prelogits_norm_p, axis=1))
        # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * prelogits_norm_loss_factor)

    def call(self, y_true, y_pred):
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(y_pred) + self.eps, ord=self.prelogits_norm_p, axis=1))
        return prelogits_norm * self.prelogits_norm_loss_factor


def prelogits_norm_loss(y_pred, from_logits=False):
    eps = 1e-4
    prelogits_norm_p = 1
    prelogits_norm_loss_factor = 5e-4
    prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(y_pred) + eps, ord=prelogits_norm_p, axis=1))
    # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * prelogits_norm_loss_factor)
    return prelogits_norm * prelogits_norm_loss_factor


class CustomModel(tf.keras.Model):
    def __init__(self, n_classes):
        super(CustomModel, self).__init__(name='my_model')
        self.total_loss = []

        self.conv_base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3),
                                                      pooling='avg')
        self.conv_base.trainable = False
        # conv_base.summary()

        bottleneck_layer_size = 512
        weight_decay = 5e-4

        self.dropout = tf.keras.layers.Dropout(0.4, seed=666)

        kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
        self.dense_bottleneck = tf.keras.layers.Dense(bottleneck_layer_size, kernel_regularizer=kernel_regularizer)
        self.prelogits = tf.keras.layers.BatchNormalization(name='Bottleneck')
        # self.prelogits = tf.keras.layers.Activation('relu')

        kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
        self.dense2 = tf.keras.layers.Dense(n_classes, kernel_regularizer=kernel_regularizer)

    def call(self, inputs, **kwargs):
        output = self.conv_base(inputs)
        output = self.dropout(output)
        output = self.dense_bottleneck(output)
        output = self.prelogits(output)
        prelogits = self.prelogits(output)
        output = self.dense2(prelogits)

        return prelogits, output

    def compute_output_shape(self, input_shape):
        print(input_shape)


def sequential_model(conv_base, weight_decay, bottleneck_layer_size, n_classes):
    model = tf.keras.Sequential()
    model.add(conv_base)
    model.add(tf.keras.layers.Dropout(0.4, seed=666))

    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    model.add(tf.keras.layers.Dense(bottleneck_layer_size, kernel_regularizer=kernel_regularizer))
    model.add(tf.keras.layers.BatchNormalization(name='Bottleneck'))
    # model.add(tf.keras.layers.Activation('relu'))

    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    model.add(tf.keras.layers.Dense(n_classes, kernel_regularizer=kernel_regularizer))

    return model


def functional_model(conv_base, input_shape, weight_decay, bottleneck_layer_size, n_classes):
    iminputs = tf.keras.Input(shape=input_shape, name='input')
    model = conv_base(iminputs)
    model = tf.keras.layers.Dropout(0.4, seed=666)(model)

    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    model = tf.keras.layers.Dense(bottleneck_layer_size, kernel_regularizer=kernel_regularizer)(model)
    prelogits = tf.keras.layers.BatchNormalization(name='Bottleneck')(model)
    # prelogits = tf.keras.layers.Activation('relu')(prelogits)

    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    output = tf.keras.layers.Dense(n_classes, kernel_regularizer=kernel_regularizer)(prelogits)
    output = tf.keras.layers.Activation('softmax')(output)

    model = tf.keras.Model(inputs=iminputs, outputs=[prelogits, output])

    return model


def custom_model(n_classes, input_shape=(28, 28, 1)):
    conv_base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    # conv_base = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    # conv_base = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    conv_base.trainable = True
    # conv_base.summary()
    #for layer in conv_base.layers[:-4]:
    #    layer.trainable = False

    bottleneck_layer_size = 512
    weight_decay = 5e-4

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # model = sequential_model(conv_base, weight_decay, bottleneck_layer_size, n_classes)  # OK
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model = functional_model(conv_base, input_shape, weight_decay, bottleneck_layer_size, n_classes)  # OK
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    return model


def scheduler(epoch):
    if epoch < 100:
        return 0.05
    elif epoch < 200:
        return 0.005
    elif epoch < 276:
        return 0.0005
    else:
        # return 0.0005 * np.exp(0.01 * (70 - epoch))
        return 0.0001


def scheduler_experiment(epoch):
    if epoch < 5:
        return 0.05
    elif epoch < 10:
        return 0.005
    elif epoch < 15:
        return 0.0005
    else:
        # return 0.0005 * np.exp(0.01 * (70 - epoch))
        return 0.0001


class TrainReports(object):
    def __init__(self):
        big_samples = 3000000
        big_epochs = 100
        tiny_samples = 1000
        tiny_epochs = 5
        k = (big_epochs-tiny_epochs)/(big_samples-tiny_samples)
        b = tiny_epochs - big_epochs*k
        y = k*100000+b
        print(y)

        self.records = {}
        self.current_key = ''

    def add_record(self, key, value):
        self.records[key] = value

    def malloc_record(self, key):
        self.records[key] = []
        self.current_key = key

    def record_cb(self, line):
        self.records[self.current_key].append(line)

    def save_record(self, save_file):
        with open(save_file, 'w') as f:
            for key, value in iteritems(self.records):
                if type(value) == list:
                    for val in value:
                        f.write('%s\n' % str(val))
                else:
                    f.write('%s: %s\n' % (key, str(value)))


def toy_model(input_shape=(28, 28, 1)):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, [3, 3], activation='relu', input_shape=(None, None, input_shape[2])),
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.GlobalMaxPool2D(),
        tf.keras.layers.Dense(32)
    ])

    return model


class Trainer(object):
    def __init__(self):
        self.params = {}

    def load_dataset(self, data_path, min_nrof_cls=10, max_nrof_cls=4000, validation_ratio=0.05):
        train_images_path, train_images_label, validation_images_path, validation_images_label = datset.load_dataset(
            data_path, min_nrof_cls=min_nrof_cls, max_nrof_cls=max_nrof_cls, validation_ratio=validation_ratio)
        num_train_images = len(train_images_path)
        num_validation_images = len(validation_images_path)
        num_classes = len(set(train_images_label))

        self.params['train_images_path'] = train_images_path
        self.params['train_images_label'] = train_images_label
        self.params['validation_images_path'] = validation_images_path
        self.params['validation_images_label'] = validation_images_label
        self.params['num_train_images'] = num_train_images
        self.params['num_validation_images'] = num_validation_images
        self.params['num_classes'] = num_classes

    def set_train_params(self, imshape=(160, 160, 3), batch_size=96, max_epochs=1):
        self.params['max_epochs'] = max_epochs
        self.params['imshape'] = imshape
        self.params['batch_size'] = batch_size
        self.params['max_epochs'] = max_epochs
        self.params['train_steps_per_epoch'] = tools.steps_per_epoch(self.params['num_train_images'], batch_size, allow_less_batsize=False)
        self.params['validation_steps_per_epoch'] = tools.steps_per_epoch(self.params['num_validation_images'], batch_size, allow_less_batsize=False)
        print('train_steps_per_epoch={}'.format(self.params['train_steps_per_epoch']))
        print('validation_steps_per_epoch={}'.format(self.params['validation_steps_per_epoch']))

        '''
        train_reports.add_record('num_classes', num_classes)
        train_reports.add_record('batch_size', batch_size)
        train_reports.add_record('initial_epochs', initial_epochs)
        train_reports.add_record('train_steps_per_epoch', train_steps_per_epoch)
        train_reports.add_record('validation_steps_per_epoch', validation_steps_per_epoch)
        train_reports.add_record('imshape', (160, 160, 3))
        '''

        imparse = datset.ImageParse(imshape=imshape)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        filenames = tf.constant(self.params['train_images_path'])
        labels = tf.constant(self.params['train_images_label'])
        train_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        train_dataset = train_dataset.map(imparse.train_parse_func)
        self.train_dataset = train_dataset.shuffle(buffer_size=min(self.params['num_train_images'], 1000),
                                              seed=tf.compat.v1.set_random_seed(666),
                                              reshuffle_each_iteration=True).batch(batch_size).repeat()  # repeat 不指定参数表示允许无穷迭代
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        filenames = tf.constant(self.params['validation_images_path'])
        labels = tf.constant(self.params['validation_images_label'])
        validation_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        validation_dataset = validation_dataset.map(imparse.validation_parse_func)
        self.validation_dataset = validation_dataset.shuffle(buffer_size=min(self.params['num_validation_images'], 1000),
                                                        seed=tf.compat.v1.set_random_seed(666),
                                                        reshuffle_each_iteration=True).batch(batch_size).repeat()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_model(self, conv_base=''):
        if conv_base == 'MobileNetV2':
            conv_base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=self.params['imshape'], pooling='avg')
        elif conv_base == 'Xception':
            conv_base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=self.params['imshape'], pooling='avg')
        elif conv_base == 'InceptionResnetV2':
            conv_base_model = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=self.params['imshape'], pooling='avg')
        else:
            conv_base_model = toy_model(self.params['imshape'])
        conv_base_model.trainable = True
        # conv_base_model.summary()
        # for layer in conv_base_model.layers[:-4]:
        #    layer.trainable = False

        bottleneck_layer_size = 512
        weight_decay = 5e-4

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # model = sequential_model(conv_base_model, self.params['imshape'], weight_decay, bottleneck_layer_size, self.params['num_classes'])  # OK
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.model = functional_model(conv_base_model, self.params['imshape'], weight_decay, bottleneck_layer_size, self.params['num_classes'])  # OK
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

    def set_record(self):
        log_dir = os.path.join('logs', tools.get_strtime() + '-n_cls:' + str(self.params['num_classes']))
        self.callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler_experiment, verbose=1),
                          tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1),
                          ]

    def set_loss_metric(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=[prelogits_norm_loss, 'sparse_categorical_crossentropy'],
                           metrics=[[], 'accuracy'])

    def fit(self):
        history = self.model.fit(self.train_dataset,
                            steps_per_epoch=self.params['train_steps_per_epoch'],
                            epochs=self.params['max_epochs'],
                            validation_data=self.validation_dataset,
                            validation_steps=self.params['validation_steps_per_epoch'],
                            callbacks=self.callbacks,
                            # workers=4,
                            # use_multiprocessing=True
                            )


if __name__ == '__main__':
    train_reports = TrainReports()
    trainer = Trainer()
    trainer.load_dataset(g_datapath)
    trainer.set_train_params(imshape=(160, 160, 3), batch_size=96, max_epochs=5)
    trainer.set_model()
    trainer.set_loss_metric()
    trainer.set_record()
    trainer.fit()


    """
    train_count = len(train_images_path)
    validation_count = len(validation_images_path)
    num_classes = len(set(train_images_label))
    batch_size = 96
    initial_epochs = 5
    train_steps_per_epoch = tools.steps_per_epoch(train_count, batch_size, allow_less_batsize=False)
    validation_steps_per_epoch = tools.steps_per_epoch(validation_count, batch_size, allow_less_batsize=False)
    print('train_steps_per_epoch={}'.format(train_steps_per_epoch))
    print('validation_steps_per_epoch={}'.format(validation_steps_per_epoch))

    train_reports.add_record('num_classes', num_classes)
    train_reports.add_record('batch_size', batch_size)
    train_reports.add_record('initial_epochs', initial_epochs)
    train_reports.add_record('train_steps_per_epoch', train_steps_per_epoch)
    train_reports.add_record('validation_steps_per_epoch', validation_steps_per_epoch)
    train_reports.add_record('imshape', (160, 160, 3))


    imparse = datset.ImageParse((160, 160, 3))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    filenames = tf.constant(train_images_path)
    labels = tf.constant(train_images_label)
    train_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    train_dataset = train_dataset.map(imparse.train_parse_func)
    train_dataset = train_dataset.shuffle(buffer_size=min(train_count, 10000), seed=tf.compat.v1.set_random_seed(666),
                              reshuffle_each_iteration=True).batch(batch_size).repeat()  # repeat 不指定参数表示允许无穷迭代
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    filenames = tf.constant(validation_images_path)
    labels = tf.constant(validation_images_label)
    validation_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    validation_dataset = validation_dataset.map(imparse.validation_parse_func)
    validation_dataset = validation_dataset.shuffle(buffer_size=min(validation_count, 10000), seed=tf.compat.v1.set_random_seed(666),
                                          reshuffle_each_iteration=True).batch(batch_size).repeat()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    model = custom_model(num_classes)
    # model = CustomModel(num_classes)
    
    train_reports.malloc_record('Model')
    model.summary(print_fn=train_reports.record_cb)

    log_dir = os.path.join('logs', tools.get_strtime() + '-n_cls:' + str(num_classes))
    callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler_experiment, verbose=1),
                 tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1),
                 ]

    optimizer = tf.keras.optimizers.Adam(lr=0.001)

    model.compile(optimizer=optimizer, loss=[prelogits_norm_loss, 'sparse_categorical_crossentropy'], metrics=[[], 'accuracy'])

    history = model.fit(train_dataset,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=initial_epochs,
                        validation_data=validation_dataset,
                        validation_steps=validation_steps_per_epoch,
                        callbacks=callbacks,
                        # workers=4,
                        # use_multiprocessing=True
                        )

    train_reports.save_record(os.path.join(log_dir, 'train_reports.txt'))
    exit(0)
    """


    print('finetune ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    initial_epochs += max(int(initial_epochs*0.7), 10)

    model.layers[1].trainable = True
    for layer in model.layers[1].layers[:-10]:
        layer.trainable = False

    optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss=[prelogits_norm_loss, 'sparse_categorical_crossentropy'],
                  metrics=[[], 'accuracy'])

    log_dir = os.path.join('logs', tools.get_strtime() + '-Finetune' + '-n_cls:' + str(num_classes))
    callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler_experiment, verbose=1),
                 tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1),
                 ]
    history = model.fit(train_dataset,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=initial_epochs,
                        validation_data=validation_dataset,
                        validation_steps=validation_steps_per_epoch,
                        callbacks=callbacks,
                        # workers=4,
                        # use_multiprocessing=True
                        )



    print('finetune ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    initial_epochs = 300

    model.layers[1].trainable = True
    for layer in model.layers[1].layers:
        layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss=[prelogits_norm_loss, 'sparse_categorical_crossentropy'],
                  metrics=[[], 'accuracy'])

    log_dir = os.path.join('logs', tools.get_strtime() + '-Finetune' + '-n_cls:' + str(num_classes))
    callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
                 tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1),
                 ]
    history = model.fit(train_dataset,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=initial_epochs,
                        validation_data=validation_dataset,
                        validation_steps=validation_steps_per_epoch,
                        callbacks=callbacks,
                        # workers=4,
                        # use_multiprocessing=True
                        )


if __name__ == '__main__2':
    def func(x, y):
        return x*y

    f = lambda x, y, z: func(x, y)+z
    a = f(1, 2, 3)
    print(a)





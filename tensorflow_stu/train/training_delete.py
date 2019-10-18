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


home_path = os.environ['HOME']

user_name = getpass.getuser()
host_name = socket.gethostname()

if user_name in ['xiajun']:
    g_datapath = os.path.join(home_path, 'res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44')
elif user_name in ['yp']:
    g_datapath = os.path.join(home_path, 'res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44_front999')
elif user_name in ['xiaj']:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    g_datapath = os.path.join(home_path, 'res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44')
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


def functional_model(conv_base, weight_decay, bottleneck_layer_size, n_classes):
    iminputs = tf.keras.Input(shape=(160, 160, 3), name='input')
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


def custom_model(n_classes):
    conv_base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3), pooling='avg')
    # conv_base = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(160, 160, 3), pooling='avg')
    # conv_base = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3), pooling='avg')
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
    model = functional_model(conv_base, weight_decay, bottleneck_layer_size, n_classes)  # OK
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


if __name__ == '__main__':
    train_reports = TrainReports()

    train_images_path, train_images_label, validation_images_path, validation_images_label = datset.load_dataset(g_datapath, min_nrof_cls=10, max_nrof_cls=40000, validation_ratio=0.05)

    train_count = len(train_images_path)
    validation_count = len(validation_images_path)
    n_classes = len(set(train_images_label))
    batch_size = 96
    initial_epochs = 5
    train_steps_per_epoch = tools.steps_per_epoch(train_count, batch_size, allow_less_batsize=False)
    validation_steps_per_epoch = tools.steps_per_epoch(validation_count, batch_size, allow_less_batsize=False)
    print('train_steps_per_epoch={}'.format(train_steps_per_epoch))
    print('validation_steps_per_epoch={}'.format(validation_steps_per_epoch))

    train_reports.add_record('n_classes', n_classes)
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
    train_dataset = train_dataset.shuffle(buffer_size=min(train_count, 1000), seed=tf.compat.v1.set_random_seed(666),
                              reshuffle_each_iteration=True).batch(batch_size).repeat()  # repeat 不指定参数表示允许无穷迭代
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    filenames = tf.constant(validation_images_path)
    labels = tf.constant(validation_images_label)
    validation_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    validation_dataset = validation_dataset.map(imparse.validation_parse_func)
    validation_dataset = validation_dataset.shuffle(buffer_size=min(validation_count, 1000), seed=tf.compat.v1.set_random_seed(666),
                                          reshuffle_each_iteration=True).batch(batch_size).repeat()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    model = custom_model(n_classes)
    # model = CustomModel(n_classes)

    train_reports.malloc_record('Model')
    model.summary(print_fn=train_reports.record_cb)

    log_dir = os.path.join('logs', tools.get_strtime() + '-n_cls:' + str(n_classes))
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


    print('finetune ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    initial_epochs += max(int(initial_epochs*0.7), 10)

    model.layers[1].trainable = True
    for layer in model.layers[1].layers[:-10]:
        layer.trainable = False

    optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss=[prelogits_norm_loss, 'sparse_categorical_crossentropy'],
                  metrics=[[], 'accuracy'])

    log_dir = os.path.join('logs', tools.get_strtime() + '-Finetune' + '-n_cls:' + str(n_classes))
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

    log_dir = os.path.join('logs', tools.get_strtime() + '-Finetune' + '-n_cls:' + str(n_classes))
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





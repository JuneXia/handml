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

if user_name in ['xiajun', 'yp']:
    g_datapath = os.path.join(home_path, 'res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44')
elif user_name in ['xiaj'] and host_name in ['ailab-server']:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # g_datapath = os.path.join(home_path, 'res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44')
    g_datapath = '/disk2/res/VGGFace2/Experiment/mtcnn_align182x182_margin44'
elif user_name in ['xiaj'] and host_name in ['ubuntu-pc']:
    g_datapath = os.path.join(home_path, 'res/mnist')
else:
    print('unkown user_name:{}'.format(user_name))
    exit(0)

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1)  # local 0.333
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config, execution_mode=tf.contrib.eager.SYNC)
print('is eager executing: ', tf.compat.v1.executing_eagerly())


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
    def __init__(self, n_classes, conv_base):
        super(CustomModel, self).__init__(name='my_model')
        self.total_loss = []

        self.conv_base = get_conv_base_model(conv_base=conv_base, imshape=(160, 160, 3))

        self.conv_base.trainable = False
        # conv_base.summary()

        bottleneck_size = 512
        weight_decay = 5e-4

        self.dropout = tf.keras.layers.Dropout(0.5, seed=666)

        kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
        self.dense_bottleneck = tf.keras.layers.Dense(bottleneck_size, kernel_regularizer=kernel_regularizer)
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


def sequential_model(conv_base, weight_decay, bottleneck_size, n_classes):
    model = tf.keras.Sequential()
    model.add(conv_base)
    model.add(tf.keras.layers.Dropout(0.5, seed=666))

    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    model.add(tf.keras.layers.Dense(bottleneck_size, kernel_regularizer=kernel_regularizer))
    model.add(tf.keras.layers.BatchNormalization(name='Bottleneck'))
    # model.add(tf.keras.layers.Activation('relu'))

    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    model.add(tf.keras.layers.Dense(n_classes, kernel_regularizer=kernel_regularizer))

    return model


def functional_model(conv_base, input_shape, weight_decay, bottleneck_size, n_classes):
    iminputs = tf.keras.Input(shape=input_shape, name='input')
    model = conv_base(iminputs)
    model = tf.keras.layers.Dropout(0.5, seed=666)(model)

    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    model = tf.keras.layers.Dense(bottleneck_size, kernel_regularizer=kernel_regularizer)(model)
    prelogits = tf.keras.layers.BatchNormalization(name='Bottleneck')(model)
    # prelogits = tf.keras.layers.Activation('relu')(prelogits)

    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    output = tf.keras.layers.Dense(n_classes, kernel_regularizer=kernel_regularizer)(prelogits)
    output = tf.keras.layers.Activation('softmax')(output)

    if datset.MULTI_OUTPUT:
        model = tf.keras.Model(inputs=iminputs, outputs=[prelogits, output])
    else:
        model = tf.keras.Model(inputs=iminputs, outputs=[output])

    return model


def custom_model(n_classes, input_shape=(28, 28, 1)):
    conv_base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    # conv_base = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    # conv_base = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    conv_base.trainable = True
    # conv_base.summary()
    #for layer in conv_base.layers[:-4]:
    #    layer.trainable = False

    bottleneck_size = 512
    weight_decay = 5e-4

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # model = sequential_model(conv_base, weight_decay, bottleneck_size, n_classes)  # OK
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model = functional_model(conv_base, input_shape, weight_decay, bottleneck_size, n_classes)  # OK
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    return model


def scheduler(epoch):
    if epoch < 3:
        return 0.1
    elif epoch < 10:
        return 0.05
    elif epoch < 20:
        return 0.005
    elif epoch < 30:
        return 0.0005
    elif epoch < 40:
        return 0.0001

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


class Reporter(object):
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

    def add_records(self, params_dict):
        for key, value in iteritems(params_dict):
            self.add_record(key, value)

    def malloc_record(self, key):
        self.records[key] = []
        self.current_key = key

    def record_cb(self, line):
        self.records[self.current_key].append(line)

    def save_record(self, save_file):
        records = list(self.records.items())
        records.sort()
        with open(save_file, 'w') as f:
            for key, value in records:
                if type(value) == list:
                    f.write('\n%s: \n' % key)
                    for val in value:
                        f.write('%s\n' % str(val))
                    f.write('\n')
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


def get_conv_base_model(conv_base, imshape):
    if conv_base == 'MobileNetV2':
        conv_base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False,
                                                            input_shape=imshape, pooling='avg')
    elif conv_base == 'Xception':
        conv_base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False,
                                                         input_shape=imshape, pooling='avg')
    elif conv_base == 'InceptionResnetV2':
        conv_base_model = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False,
                                                                  input_shape=imshape, pooling='avg')
    elif conv_base == 'DenseNet201':
        conv_base_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False,
                                                            input_shape=imshape, pooling='avg')
    elif conv_base == 'ResNet50':
        conv_base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False,
                                                         input_shape=imshape, pooling='avg')
    else:
        conv_base_model = toy_model(imshape)

    return conv_base_model


class Trainer(object):
    def __init__(self):
        self.params = {}  # TODO： 将所有的参数都放在着一个字典里可能不太合适，感觉还是多分几个字典比较好。
        self.params['executing_eagerly'] = tf.executing_eagerly()
        self.datas = {}

    def get_params(self):
        return self.params

    def get_logdir(self):
        return self.log_dir

    def load_dataset(self, data_path, min_nrof_cls=10, max_nrof_cls=4000, validation_ratio=0.1):
        train_images_path, train_images_label, validation_images_path, validation_images_label = datset.load_dataset(
            data_path, min_nrof_cls=min_nrof_cls, max_nrof_cls=max_nrof_cls, validation_ratio=validation_ratio)
        num_train_images = len(train_images_path)
        num_validation_images = len(validation_images_path)
        num_classes = len(set(train_images_label))

        self.datas['train_images_path'] = train_images_path
        self.datas['train_images_label'] = train_images_label
        self.datas['validation_images_path'] = validation_images_path
        self.datas['validation_images_label'] = validation_images_label

        self.params['data_path'] = data_path
        self.params['min_nrof_cls'] = min_nrof_cls
        self.params['max_nrof_cls'] = max_nrof_cls
        self.params['validation_ratio'] = validation_ratio
        self.params['num_train_images'] = num_train_images
        self.params['num_validation_images'] = num_validation_images
        self.params['num_classes'] = num_classes

    def set_train_params(self, imshape=(160, 160, 3), batch_size=96, max_epochs=1):
        self.params['max_epochs'] = max_epochs
        self.params['imshape'] = imshape
        self.params['batch_size'] = batch_size
        self.params['max_epochs'] = max_epochs
        if user_name in ['xiaj'] and host_name in ['ailab-server']:
            # TODO: ailab训练速度慢，而且facenet源码epoch_size也是等于1000，故这里暂且用1000吧。
            self.params['train_steps_per_epoch'] = 1000
        else:
            self.params['train_steps_per_epoch'] = tools.steps_per_epoch(self.params['num_train_images'], batch_size, allow_less_batsize=False)
        self.params['validation_steps_per_epoch'] = tools.steps_per_epoch(self.params['num_validation_images'], batch_size, allow_less_batsize=False)
        print('train_steps_per_epoch={}'.format(self.params['train_steps_per_epoch']))
        print('validation_steps_per_epoch={}'.format(self.params['validation_steps_per_epoch']))

        imparse = datset.ImageParse(imshape=imshape, n_classes=self.params['num_classes'])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        filenames = tf.constant(self.datas['train_images_path'])
        labels = tf.constant(self.datas['train_images_label'])
        train_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        train_dataset = train_dataset.map(imparse.train_parse_func)
        self.train_dataset = train_dataset.shuffle(buffer_size=min(self.params['num_train_images'], 1000),
                                              seed=tf.compat.v1.set_random_seed(666),
                                              reshuffle_each_iteration=True).batch(batch_size).repeat().prefetch(buffer_size=-1)  # repeat 不指定参数表示允许无穷迭代
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        filenames = tf.constant(self.datas['validation_images_path'])
        labels = tf.constant(self.datas['validation_images_label'])
        validation_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        validation_dataset = validation_dataset.map(imparse.validation_parse_func)
        self.validation_dataset = validation_dataset.shuffle(buffer_size=min(self.params['num_validation_images'], 1000),
                                                        seed=tf.compat.v1.set_random_seed(666),
                                                        reshuffle_each_iteration=True).batch(batch_size).repeat().prefetch(buffer_size=-1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_model(self, conv_base='toy_model', custom_model=False):
        self.params['conv_base'] = conv_base
        self.params['bottleneck_size'] = 512
        self.params['weight_decay'] = 5e-4

        if custom_model:
            self.model = CustomModel(self.params['num_classes'], conv_base)
            imshape = self.params['imshape']
            self.model.build(input_shape=(None, imshape[0], imshape[1], imshape[2]))
        else:
            conv_base_model = get_conv_base_model(conv_base, self.params['imshape'])
            conv_base_model.trainable = True

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # model = sequential_model(conv_base_model, self.params['imshape'], self.params['weight_decay'], self.params['bottleneck_size'], self.params['num_classes'])  # OK
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.model = functional_model(conv_base_model, self.params['imshape'], self.params['weight_decay'],
                                          self.params['bottleneck_size'], self.params['num_classes'])  # OK
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        print('create model finish')

    def set_record(self):
        self.save_basename = tools.strcat([tools.get_strtime(), self.params['conv_base'], 'n_cls:' + str(self.params['num_classes'])], cat_mark='-')
        self.log_dir = os.path.join('logs', self.save_basename)
        self.model_save_dir = os.path.join('save_model', self.save_basename)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        model_ckpt_dir = os.path.join(self.model_save_dir, 'ckpt')
        if not os.path.exists(model_ckpt_dir):
            os.makedirs(model_ckpt_dir)
        self.model_ckpt_dir = os.path.join(model_ckpt_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')

        self.callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
                          tf.keras.callbacks.TensorBoard(self.log_dir, histogram_freq=0),  # TODO: 对于非eager模式，histogram_freq不能设为1
                          # tf.keras.callbacks.ModelCheckpoint(self.model_ckpt_dir, verbose=1, save_best_only=True, save_freq=2)  # save_freq>1时 运行时会崩溃。
                          ]

    def set_loss_metric(self, losses=[], metrics=[]):
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=self.optimizer,
                           loss=losses,
                           metrics=metrics)

        self.params['losses'] = losses
        self.params['metrics'] = metrics

    def fit(self):
        history = self.model.fit(self.train_dataset,
                            steps_per_epoch=self.params['train_steps_per_epoch'],
                            epochs=self.params['max_epochs'],
                            validation_data=self.validation_dataset,
                            validation_steps=self.params['validation_steps_per_epoch'],
                            callbacks=self.callbacks,
                            validation_freq=5,
                            workers=4,
                            use_multiprocessing=True
                            )

    def custom_fit(self):
        for batch, (images, labels) in enumerate(self.train_dataset):
            # self.global_step.assign_add(1)
            with tf.GradientTape() as t:
                outputs = self.model(images)
                if type(outputs) != np.ndarray:
                    outputs = outputs.numpy()
                labels = labels.reshape((-1, 1))
                loss_step = self.loss_func(labels, outputs)

    def save_model(self):
        # TODO: 临时写法
        try:
            self.model.save(os.path.join(self.model_save_dir, 'Model.h5'))
            checkpoint_path = os.path.join(self.model_save_dir, 'ckpt')
            self.model.save_weights(checkpoint_path)
        except:
            checkpoint_path = os.path.join(self.model_save_dir, 'ckpt')
            self.model.save_weights(checkpoint_path)

    def finetune(self):
        conv_base_model = self.model.layers[1]
        for layer in conv_base_model.layers[:-4]:
            layer.trainable = False


if __name__ == '__main__':
    reporter = Reporter()
    trainer = Trainer()
    trainer.load_dataset(g_datapath, min_nrof_cls=10, max_nrof_cls=4000, validation_ratio=0.02)
    # trainer.load_dataset(g_datapath, min_nrof_cls=10, max_nrof_cls=4000, validation_ratio=0.1)
    # trainer.set_train_params(imshape=(28, 28, 1), batch_size=96, max_epochs=2)
    trainer.set_train_params(imshape=(160, 160, 3), batch_size=96, max_epochs=276)
    trainer.set_model('ResNet50', custom_model=False)  # InceptionResnetV2, MobileNetV2, Xception, ResNet50
    trainer.set_record()
    if datset.MULTI_OUTPUT:
        trainer.set_loss_metric(losses=[prelogits_norm_loss, 'sparse_categorical_crossentropy'], metrics=[[], 'accuracy'])
    else:
        # trainer.set_loss_metric(losses=['sparse_categorical_crossentropy'], metrics=['accuracy'])
        trainer.set_loss_metric(losses=['categorical_crossentropy'], metrics=['accuracy'])
    reporter.add_records(trainer.get_params())

    trainer.fit()
    # trainer.custom_fit()
    trainer.save_model()
    reporter.save_record(os.path.join(trainer.get_logdir(), 'params.txt'))


# MovingAverage在Tensorflow2中可能需要在自定义训练中才能实现，或者自己继承tf的相关类来实现。
# tf.moving_average_variables

# 单输出模型似乎比多输出模型训练快


if __name__ == '__main__2':
    def func(x, y):
        return x*y

    f = lambda x, y, z: func(x, y)+z
    a = f(1, 2, 3)
    print(a)


def dup_sampling_check(info):
    if len(info) > 200:
        return False

    marks = ['happyjuzi_mainland', 'happyjuzi_HongkongTaiwan']
    # samples_info = []
    for dat in info:
        if 'VGGFace2' in dat[-1]:
            # 不对VGGFace2数据集进行重复抽样
            return False

        findit = False
        for mark in marks:
            if mark in dat[-1]:
                findit = True
                # samples_info.append(dat)
                break
        if not findit:
            raise Exception('Only support dataset: {}'.format(marks))

    # samples_info = np.array(samples_info)
    # return samples_info
    return True


if __name__ == '__main__3':  # 加载facenet训练所需要的数据集
    train_dataset = [{'root_path': '/disk1/home/xiaj/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44',
                      'csv_file': '/disk1/home/xiaj/res/face/VGGFace2/Experiment/VGGFace2_cleaned_with_happyjuzi_mainland.csv'},
                     {'root_path': '/disk1/home/xiaj/res/face/GC-WebFace/Experiment/mtcnn_align182x182_margin44_happyjuzi_mainland_cleaning',
                      'csv_file': '/disk1/home/xiaj/res/face/GC-WebFace/Experiment/happyjuzi_mainland_cleaned.csv'},
                     ]

    # datset.make_feedata(train_dataset, 'tmp2.csv', title_line='train_val,label,person_name,image_path\n', filter_cb=dup_sampling_check)
    datset.load_feedata('tmp2.csv')



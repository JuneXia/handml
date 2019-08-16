import os
import sys

from utils import util

network_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(network_path)
from network import lenet3 as net
from network import siamese as Model
from datasets import dataset

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import cv2


from tensorflow.examples.tutorials import mnist

# from tensorflow.data import Dataset


# Change Mark: 相对于lenet2.py，本次实验必须使用eager模式，否则会报错
tf.enable_eager_execution()
print('is eager executing: ', tf.executing_eagerly())


if __name__ == '__main__1':
    (mnist_images, mnist_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    mnist_labels = to_categorical(mnist_labels, 10)
    mnist_images = np.expand_dims(mnist_images, axis=3) / 255
    mnist_labels = mnist_labels.astype(np.float)

    model = tf.keras.Sequential([
        net.LeNet(),

        # 这种先用sigmoid再用softmax的方法也可以，但是实测准确率没有直接用softmax高
        # ****************************************
        # layers.Dense(10, activation='sigmoid')
        # layers.Activation('softmax')
        # ****************************************

        # OK, GOOD
        # ****************************************
        layers.Dense(10, activation='softmax')
        # ****************************************
    ])

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(mnist_images, mnist_labels, batch_size=32, epochs=5)


class ClassificationNet(object):
    def __init__(self, input_hold, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()

        self.prelogits = embedding_net(input_hold)

        weight_decay = 5e-4
        self.prelogits = self.prelogits
        self.n_classes = n_classes
        self.embeddings = tf.nn.l2_normalize(self.prelogits, 1, 1e-10, name='embeddings')

        self.logits = slim.fully_connected(self.prelogits, n_classes, activation_fn=None,
                                      weights_initializer=slim.initializers.xavier_initializer(),
                                      weights_regularizer=slim.l2_regularizer(weight_decay),
                                      scope='Logits', reuse=False)


def _parse_function(filename, label):
    imsize = (28, 28)
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, imsize)
    image = np.expand_dims(image_resized, axis=3) / 255
    return image[0], label


def sequence_dataset(datas_path, labels, num_class, batch_size=1, buffer_size=1000, one_hot=True):
    if one_hot:
        labels = to_categorical(labels, num_class).astype(np.float)

    filenames = tf.constant(datas_path)
    labels = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=buffer_size, seed=666, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()  # dataset.repeat()不传参数表示迭代无数轮。

    # iterator = dataset.make_initializable_iterator()
    # next_element = iterator.get_next()

    return dataset


def sequence_dataset_holder(datas_placeholder, labels_placeholder, num_class, batch_size=1, one_hot=True):
    #if one_hot:
    #    images_label = to_categorical(images_label, num_class).astype(np.float)

    #filenames = tf.constant(images_path)
    #labels = tf.constant(images_label)
    dataset = tf.data.Dataset.from_tensor_slices((datas_placeholder, labels_placeholder))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=10000, seed=666, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()  # dataset.repeat()不传参数表示迭代无数轮。

    iterator = dataset.make_initializable_iterator()
    dataset = iterator.get_next()

    return dataset


if __name__ == '__main__2':
    data_path = '/home/xiajun/res/MNIST_data/train'
    images_path, images_label = util.get_dataset(data_path)
    num_class = len(set(images_label))
    batch_size = 100
    dataset = sequence_dataset(images_path, images_label, num_class, batch_size=batch_size)

    model = tf.keras.Sequential([
        net.LeNet(),
        layers.Dense(num_class, activation='softmax')
    ])

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(dataset, epochs=10, steps_per_epoch=len(images_path) // batch_size + 1)



if __name__ == '__main__3':
    data_path = '/home/xiajun/res/MNIST_data/train'
    images_path, images_label = util.get_dataset(data_path)
    num_class = len(set(images_label))
    batch_size = 100

    datas_placeholder = tf.placeholder(tf.string, shape=[None])
    labels_placeholder = tf.placeholder(tf.float32, [None, num_class])
    dataset = sequence_dataset_holder(datas_placeholder, labels_placeholder, num_class, batch_size=batch_size)

    model = tf.keras.Sequential([
        net.LeNet(),
        layers.Dense(num_class, activation='softmax')
    ])

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # >> 发现model两次fit也是可以的，这是否可以先sess.run得到数据，然后再将该数据送给model.fit
    """
    model.fit(dataset, epochs=5, steps_per_epoch=len(images_path) // batch_size + 1)
    print('~~~~~~~~~~~~~~~~~')
    model.fit(dataset, epochs=5, steps_per_epoch=len(images_path) // batch_size + 1)
    """




class DataIterator(object):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        super(DataIterator, self).__init__()
        self.count = 0
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == len(self.dataset):
            raise StopIteration

        datas = []
        start_index = min(len(self.dataset), self.count)
        end_index = min(len(self.dataset), self.count + self.batch_size)

        # datas = self.dataset[start_index:end_index]  # 如果是SiameseDataset则无法使用start:end这种索引方式，不过下面这个方式是通用的。

        for i, index in enumerate(range(start_index, end_index)):
            data = self.dataset[index]
            datas.append(data)
            self.count += 1
        datas = np.array(datas)

        return datas


# 直接调用DataIterator的话，跟for循环迭代列表没什么两样
# 本示例只是验证一下DataIterator代码是否能用。
if __name__ == '__main__DataIterator':
    data_path = '/home/xiajun/res/MNIST_data/train'
    images_path, images_label = util.get_dataset(data_path)
    num_class = len(set(images_label))
    batch_size = 100
    # dataset = sequence_dataset(images_path, images_label, num_class, batch_size=batch_size)
    dataiter = DataIterator(images_path, batch_size=10)

    for data in dataiter:
        print(data)

    for data in dataiter:
        print(data)


# TODO: 试试继承tf.data.Dataset能不能实现迭代。
class SiameseDataset1(tf.data.Dataset):
    def __init__(self):
        super(SiameseDataset1, self).__init__()


class SiameseDataset(object):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, datas, labels, is_train=True):
        super(SiameseDataset, self).__init__()
        self.datas = datas
        self.labels = labels

        self.train = is_train
        # self.transform = self.mnist_dataset.transform

        if self.train:
            self.labels_set = set(self.labels)
            self.label_to_indices = {label: np.where(self.labels == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.datas[index], self.labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.datas[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        """
        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        """
        return img1, img2, target

    def __len__(self):
        return len(self.datas)


class Dataset(object):
    """
    普通数据集
    """

    def __init__(self, datas, labels, batch_size=1, is_train=True, one_hot=True):
        super(Dataset, self).__init__()
        self.count = 0
        self.datas = datas
        self.labels = labels
        self.batch_size = batch_size
        self.num_class = len(set(labels))
        self.one_hot = one_hot

        self.train = is_train
        # self.transform = self.mnist_dataset.transform

        """
        if self.train:
            self.labels_set = set(self.labels)
            self.label_to_indices = {label: np.where(self.labels == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs
        """

    """
    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.datas[index], self.labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.datas[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        '''
        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        '''
        return img1, img2, target
    """

    def __iter__(self):
        """这个函数并不会被调用
        :return:
        """
        self.count = 0
        return self

    def __next__(self):
        if self.count == len(self.datas):
            self.count = 0
            # raise StopIteration

        datas = []
        labels = []
        start_index = min(len(self.datas), self.count)
        end_index = min(len(self.datas), self.count + self.batch_size)

        for i, index in enumerate(range(start_index, end_index)):
            data = self.datas[index]
            label = self.labels[index]
            image = cv2.imread(data).astype(np.float32)
            image /= 127.5
            datas.append(image)
            labels.append(label)
            self.count += 1
        datas = np.array(datas)
        if self.one_hot:
            labels = to_categorical(labels, self.num_class)
        else:
            labels = np.array(labels)

        return datas, labels

    def __len__(self):
        """这个函数并不会被调用
        :return:
        """
        return len(self.datas)

class DataGenerator_my(object):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        super(DataGenerator, self).__init__()
        self.count = 0
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == len(self.dataset):
            raise StopIteration

        datas = []
        start_index = min(len(self.dataset), self.count)
        end_index = min(len(self.dataset), self.count + self.batch_size)

        # datas = self.dataset[start_index:end_index]  # 如果是SiameseDataset则无法使用start:end这种索引方式，不过下面这个方式是通用的。

        for i, index in enumerate(range(start_index, end_index)):
            data = self.dataset[index]

            datas.extend([cv2.imread(data[0]), cv2.imread(data[1])])
            self.count += 1
        datas = np.array(datas)

        return datas

if __name__ == '__main__4':
    data_path = '/home/xiajun/res/MNIST_data/train'
    images_path, images_label = util.get_dataset(data_path)
    num_class = len(set(images_label))
    batch_size = 100

    dataset = SiameseDataset(images_path, images_label)

    dataset = DataIterator(dataset, batch_size=11)

    """
    for i, data in enumerate(dataset):
        print(i, data.shape)
        if i == 180:
            print('debug')
    """



    # >>> Siamese数据可以迭代了，下面该考虑如何将数据输入到网络进行训练了。
    # >>> 不过再这之前，我们应该先了解一下普通的分类网络如果不使用model.fit的话，该如何feed数据。

    # datas_placeholder = tf.placeholder(tf.string, [None])  # tf.placeholder不支持eager模式
    # labels_placeholder = tf.placeholder(tf.float32, [None, num_class])
    datas_placeholder = tf.keras.Input(dtype=tf.string, shape=[None])
    labels_placeholder = tf.keras.Input(dtype=tf.float32, shape=[None])

    # tf.data.Dataset.from_tensor_slices是为了加载全部数据路径，从而在减少内存开销的，一般不用batch数据作为它的输入
    # 我们这里实现的Siamese数据迭代方式也是先加载全部数据，然后按照自定义的需求产生batch，这相当于是实现了tf.data.Dataset.from_tensor_slices的功能，所以它和tf.data.Dataset.from_tensor_slices是平级的。
    # 我们的Siamese迭代器是产生batch数据路径，
    # 1. 先得到batch图片，然后送入网络 >>> 效率低
    # 2. 启进程用队列产生batch图片，训练单独用一个进程。tensorflow有队列实现机制
    # 3. 使用generator生成图片，然后yield通过fit_gernerator给网络
    # 方法2主键被抛弃，看来只有方法3最合适了，接下来先自己实现读图和数据增强吧。
    dataset = tf.data.Dataset.from_tensor_slices((datas_placeholder, labels_placeholder))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=10000, seed=666, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()  # dataset.repeat()不传参数表示迭代无数轮。

    # iterator = dataset.make_initializable_iterator()

    with tf.Session() as sess:
        for i, data in enumerate(dataset):  # Iterator方式只在eager模式时支持
            print(i, data.shape)
            if i == 180:
                print('debug')
            img1, img2, label = data
            sess.run(iterator.initializer, feed_dict={datas_placeholder: img2,
                                                      labels_placeholder: label})


if __name__ == '__main__5':
    data_path = '/home/xiajun/res/MNIST_data/train'
    images_path, images_label = util.get_dataset(data_path)
    num_class = len(set(images_label))
    batch_size = 110

    # dataset = SiameseDataset(images_path, images_label)
    dataset = Dataset(images_path, images_label, batch_size=batch_size)

    model = tf.keras.Sequential([
        net.LeNet(),
        layers.Dense(num_class, activation='softmax')
    ])

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # >>> next 尝试一下方法:
    # >> model.fit_generator()
    model.fit_generator(dataset, epochs=5, steps_per_epoch=len(images_path) // batch_size + 1)

    # >> model.train_on_batch()
    # model.train_on_batch(dataset)

    # >> model.train_function


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, datas, labels, shape=(), batch_size=1, one_hot=True, shuffle=True):
        super(DataGenerator, self).__init__()
        self.count = 0
        self.datas = datas
        self.labels = labels
        self.shape = shape
        self.batch_size = batch_size
        self.num_class = len(set(labels))
        self.shuffle = shuffle
        self.one_hot = one_hot

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.datas) / self.batch_size))

    def __getitem__(self, index):
        print(index)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_data = self.datas[indexes]
        batch_label = self.labels[indexes]

        X, y = self.data_generation(batch_data, batch_label)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.datas))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_data, batch_label):
        X = np.empty((self.batch_size, *self.shape))
        y = np.empty((self.batch_size), dtype=int)

        for i, (data, label) in enumerate(zip(batch_data, batch_label)):
            X[i, ] = cv2.imread(data)
            y[i] = self.labels[label]

        return X, keras.utils.to_categorical(y, num_classes=self.num_class)



if __name__ == '__main__6':
    data_path = '/home/xiajun/res/MNIST_data/train'
    images_path, images_label = util.get_dataset(data_path)
    num_class = len(set(images_label))
    batch_size = 110

    # dataset = SiameseDataset(images_path, images_label)
    # dataset = Dataset(images_path, images_label, batch_size=batch_size)
    dataset = DataGenerator(images_path, images_label, shape=(28, 28, 3), batch_size=32)

    model = tf.keras.Sequential([
        net.LeNet(),
        layers.Dense(num_class, activation='softmax')
    ])

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # >>> next 尝试一下方法:
    # >> model.fit_generator()
    model.fit_generator(dataset, epochs=5, steps_per_epoch=len(images_path) // batch_size + 1)

    # >>> 先把从网上找到的继承自keras.utils的generator调通，然后再开始siamese
    # >>> 写siamese训练，这个使用lenet3网络恐怕不行，可以尝试用lenet4试试。
    # >>> 完了再试试将lenet3作为网络其中一部分，siamese部分重新定义



if __name__ == '__main__':
    data_path = '/home/xiajun/res/MNIST_data/train'
    images_path, images_label = util.get_dataset(data_path)
    num_class = len(set(images_label))
    batch_size = 110

    # dataset = SiameseDataset(images_path, images_label)
    # dataset = Dataset(images_path, images_label, batch_size=batch_size)
    dataset = DataGenerator(images_path, images_label, shape=(28, 28, 3), batch_size=32)

    model = tf.keras.Sequential()
    model.add(net.LeNet())
    model.add(Model.SiameseNet())
    model = Model.SiameseNet(embedding_net)
    model = Model.ContrastiveLoss(1.0)

    model = tf.keras.Sequential([
        model
    ])

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(dataset, epochs=5, steps_per_epoch=len(images_path) // batch_size + 1)

    print('debug')



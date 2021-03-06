import os
import math
import random
import numpy as np
import tensorflow as tf
import itertools

class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths


def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    print(path_exp, path)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset


def split_dataset(dataset, split_ratio, min_nrof_images_per_class, mode):
    if mode=='SPLIT_CLASSES':
        nrof_classes = len(dataset)
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)
        split = int(round(nrof_classes*(1-split_ratio)))
        train_set = [dataset[i] for i in class_indices[0:split]]
        test_set = [dataset[i] for i in class_indices[split:-1]]
    elif mode=='SPLIT_IMAGES':
        train_set = []
        test_set = []
        for cls in dataset:
            paths = cls.image_paths
            np.random.shuffle(paths)
            nrof_images_in_class = len(paths)
            split = int(math.floor(nrof_images_in_class*(1-split_ratio)))
            if split==nrof_images_in_class:
                split = nrof_images_in_class-1
            if split>=min_nrof_images_per_class and nrof_images_in_class-split >= 1:
                train_set.append(ImageClass(cls.name, paths[:split]))
                test_set.append(ImageClass(cls.name, paths[split:]))
            else:
                raise ValueError('TODO: what happened!' % mode)
    else:
        raise ValueError('Invalid train/test split mode "%s"' % mode)
    return train_set, test_set


def get_image_paths_and_labels(dataset, shuffle=False):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)

    if shuffle:
        image_labels = np.vstack((image_paths_flat, labels_flat)).transpose()
        np.random.shuffle(image_labels)
        image_paths_flat = image_labels[:, 0].tolist()
        labels_flat = image_labels[:, 1].astype(np.int32).tolist()

    return image_paths_flat, labels_flat


def load_data(data_dir, validation_set_split_ratio=0.05, min_nrof_val_images_per_class=0):
    seed = 666
    np.random.seed(seed=seed)
    random.seed(seed)
    dataset = get_dataset(data_dir)
    #if filter_filename is not None:
    #    dataset = filter_dataset(dataset, os.path.expanduser(filter_filename),
    #                             filter_percentile, filter_min_nrof_images_per_class)

    if validation_set_split_ratio > 0.0:
        train_set, val_set = split_dataset(dataset, validation_set_split_ratio,
                                                   min_nrof_val_images_per_class, 'SPLIT_IMAGES')
    else:
        train_set, val_set = dataset, []

    return train_set, val_set


def get_control_flag(control, field):
    return tf.equal(tf.mod(tf.floor_div(control, field), 2), 1)

# 1: Random rotate 2: Random crop  4: Random flip  8:  Fixed image standardization  16: Flip
RANDOM_ROTATE = 1
RANDOM_CROP = 2
RANDOM_FLIP = 4
FIXED_STANDARDIZATION = 8
FLIP = 16
def _parse_function(filename, label):
    shape = [28, 28, 1]
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string, shape[2])
    # image_tensor = tf.convert_to_tensor(image_decoded, dtype=tf.float32)

    # >>>>在不知道图像尺寸的情况下，是不能乱set_shape的，仿照facenet.py中创建pipeline的方法写。
    """
    image = tf.cond(get_control_flag(control[0], RANDOM_ROTATE),
                    lambda: tf.py_func(random_rotate_image, [image], tf.uint8),
                    lambda: tf.identity(image))
    image = tf.cond(get_control_flag(control[0], RANDOM_CROP),
                    lambda: tf.random_crop(image, image_size + (3,)),
                    lambda: tf.image.resize_image_with_crop_or_pad(image, image_size[0], image_size[1]))
    image = tf.cond(get_control_flag(control[0], RANDOM_FLIP),
                    lambda: tf.image.random_flip_left_right(image),
                    lambda: tf.identity(image))
    image = tf.cond(get_control_flag(control[0], FIXED_STANDARDIZATION),
                    lambda: (tf.cast(image, tf.float32) - 127.5) / 128.0,
                    lambda: tf.image.per_image_standardization(image))
    image = tf.cond(get_control_flag(control[0], FLIP),
                    lambda: tf.image.flip_left_right(image),
                    lambda: tf.identity(image))
    # pylint: disable=no-member
    image.set_shape(image_size + (3,))
    images.append(image)
    """

    image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, shape[0], shape[1])
    return image_resized, label


class DataSet(object):
    def __init__(self, data_path, batch_size=1, repeat=1, buffer_size=10000):
        super(DataSet, self).__init__()

        train_set, val_set = load_data(data_path)
        self.n_classes = len(train_set)
        train_image_list, train_label_list = get_image_paths_and_labels(train_set, shuffle=True)
        self.epoch_size = math.ceil(len(train_image_list) / batch_size)  # 迭代一轮所需要的训练次数

        filenames = tf.constant(train_image_list)
        labels = tf.constant(train_label_list)

        # 此时dataset中的一个元素是(filename, label)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

        # 此时dataset中的一个元素是(image_resized, label)
        dataset = dataset.map(_parse_function)

        # 此时dataset中的一个元素是(image_resized_batch, label_batch)
        # dataset = dataset.shuffle(buffer_size=1000).batch(32).repeat(10)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=tf.set_random_seed(666), reshuffle_each_iteration=True).batch(batch_size).repeat(repeat)

        iterator = dataset.make_one_shot_iterator()
        self.get_next = iterator.get_next()


class FaceDataset:
    def __init__(self, path, train=True, transform=None):
        train_set, valid_set = load_data(path)

        self.train = train
        self.transform = transform
        self.train_data, self.train_labels = get_image_paths_and_labels(train_set)
        self.test_data, self.test_labels = get_image_paths_and_labels(valid_set)

        self.train_data = np.array(self.train_data)
        self.train_labels = np.array(self.train_labels)
        self.test_data = np.array(self.test_data)
        self.test_labels = np.array(self.test_labels)

    def __len__(self):
        return len(self.train_data)


import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler


class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(self.train_labels == label)[0]
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
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
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
        return len(self.mnist_dataset)


class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


class DataIterator:
    def __init__(self, dataset, batch_size=1, shuffle=False):
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
        for i, index in enumerate(range(start_index, end_index)):
            data = self.dataset[index]
            datas.append(data)
            self.count += 1
        datas = np.array(datas)

        return datas


if __name__ == '__main__':
    data_path = '/home/xiaj/res/mnist'
    train_set, val_set = load_data(data_path)
    train_image_list, train_label_list = get_image_paths_and_labels(train_set)

    filenames = tf.constant(train_image_list)
    labels = tf.constant(train_label_list)

    # 此时dataset中的一个元素是(filename, label)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # 此时dataset中的一个元素是(image_resized, label)
    dataset = dataset.map(_parse_function)

    # 此时dataset中的一个元素是(image_resized_batch, label_batch)
    # dataset = dataset.shuffle(buffer_size=1000).batch(32).repeat(10)
    dataset = dataset.shuffle(buffer_size=1000).batch(128).repeat(5)

    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        try:
            count = 0
            while True:
                count += 1
                image, label = sess.run(one_element)
                print(count, image.shape, label)
        except tf.errors.OutOfRangeError:
            print('end!')


if __name__ == "__main__2":
    data_path = '/home/xiajun/res/face/VGGFace2/Experiment/train_mtcnn_align160x160_margin32'
    VGGFaceDataset = FaceDataset(data_path)
    siamese_dataset = SiameseMNIST(VGGFaceDataset)

    data_iterator = DataIterator(siamese_dataset, batch_size=10)

    image_batch = tf.placeholder(tf.string)
    label_batch = tf.placeholder(tf.int32)

    for data in data_iterator:
        print(data)

    print('debug')

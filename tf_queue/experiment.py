"""
import tensorflow as tf

if __name__ == "__main__2":
    with tf.Session() as sess:
        index_queue = tf.train.range_input_producer(10, num_epochs=None,
                                                    shuffle=False, seed=None, capacity=32)
        dequeue_op = index_queue.dequeue_many(15, 'index_dequeue')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 注意，先调用这个函数来启动所有的queue

        while True:
            arr = sess.run(dequeue_op)
            print(arr)
"""

"""
参考文献：https://zhuanlan.zhihu.com/p/31876710
import tensorflow as tf

with tf.Session() as sess:
    qr = tf.RandomShuffleQueue(capacity=8, min_after_dequeue=2, dtypes=[tf.uint8])
    en_qr = qr.enqueue_many([[1,2,3,4,5,6,7,8]])
    sess.run(en_qr)
    for index in range(60):
        de_qr = qr.dequeue()
        res = sess.run(de_qr)  # 当队列中没有元素，或者元素个数少于min_after_dequeue时，队列会阻塞。
        print(res)
"""


"""
# 参考文献：https://zhuanlan.zhihu.com/p/31876710
import tensorflow as tf

with tf.Session() as sess:
    qr = tf.FIFOQueue(capacity=3, dtypes=[tf.uint8, tf.float64],shapes=((),()))

    en_qr = qr.enqueue_many([[1, 2, 3,], [9.9, 8.8, 7.7,]])
    sess.run(en_qr)
    print('queue size: ', qr.size().eval())
    de_qr = qr.dequeue_many(3)
    res = sess.run(de_qr)
    print(res)

    en_qr = qr.enqueue_many([[1, 2, ], [9.9, 8.8, ]])
    sess.run(en_qr)
    print('queue size: ', qr.size().eval())
    de_qr = qr.dequeue_many(3)  # 此时qr队列中此时只有2个元素长度，所以一次出队3个元素会阻塞，直到有新的元素入队。
    res = sess.run(de_qr)
    print(res)

    en_qr = qr.enqueue_many([[1, 2, 3, 4], [9.9, 8.8, 7.7, 6.6]])
    sess.run(en_qr)  # qr队列容量capacity只有3，所以一次入队4个元素会导致阻塞，指导有元素从队列中出队。
    print('queue size: ', qr.size().eval())
    de_qr = qr.dequeue_many(3)
    res = sess.run(de_qr)
    print(res)
"""

import os
import math
import random
import h5py
import numpy as np
from scipy import misc
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    #plt.plot(bin_centers, cdf)
    threshold = np.interp(percentile*0.01, cdf, bin_centers)
    return threshold

def get_control_flag(control, field):
    return tf.equal(tf.mod(tf.floor_div(control, field), 2), 1)


def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

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

def filter_dataset(dataset, data_filename, percentile, min_nrof_images_per_class):
    with h5py.File(data_filename,'r') as f:
        distance_to_center = np.array(f.get('distance_to_center'))
        label_list = np.array(f.get('label_list'))
        image_list = np.array(f.get('image_list'))
        distance_to_center_threshold = find_threshold(distance_to_center, percentile)
        indices = np.where(distance_to_center>=distance_to_center_threshold)[0]
        filtered_dataset = dataset
        removelist = []
        for i in indices:
            label = label_list[i]
            image = image_list[i]
            if image in filtered_dataset[label].image_paths:
                filtered_dataset[label].image_paths.remove(image)
            if len(filtered_dataset[label].image_paths)<min_nrof_images_per_class:
                removelist.append(label)

        ix = sorted(list(set(removelist)), reverse=True)
        for i in ix:
            del(filtered_dataset[i])

    return filtered_dataset

def load_data(data_dir, validation_set_split_ratio=0.05, min_nrof_val_images_per_class=0, filter_filename=None, filter_percentile=100, filter_min_nrof_images_per_class=0):
    seed = 666
    np.random.seed(seed=seed)
    random.seed(seed)
    dataset = get_dataset(data_dir)
    if filter_filename is not None:
        dataset = filter_dataset(dataset, os.path.expanduser(filter_filename),
                                 filter_percentile, filter_min_nrof_images_per_class)

    if validation_set_split_ratio > 0.0:
        train_set, val_set = split_dataset(dataset, validation_set_split_ratio,
                                                   min_nrof_val_images_per_class, 'SPLIT_IMAGES')
    else:
        train_set, val_set = dataset, []

    return train_set, val_set


# 1: Random rotate 2: Random crop  4: Random flip  8:  Fixed image standardization  16: Flip
RANDOM_ROTATE = 1
RANDOM_CROP = 2
RANDOM_FLIP = 4
FIXED_STANDARDIZATION = 8
FLIP = 16


def create_input_pipeline(input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder):
    images_and_labels_list = []
    for _ in range(nrof_preprocess_threads):
        filenames, label, control = input_queue.dequeue()
        images = []
        for filename in tf.unstack(filenames):
            file_contents = tf.read_file(filename)
            image = tf.image.decode_image(file_contents, 3)
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
        images_and_labels_list.append([images, label])

    image_batch, label_batch = tf.train.batch_join(
        images_and_labels_list, batch_size=batch_size_placeholder,
        shapes=[image_size + (3,), ()], enqueue_many=True,
        capacity=4 * nrof_preprocess_threads * 100,
        allow_smaller_final_batch=True)

    return image_batch, label_batch


class QueueTrain:
    def __init__(self, train_set):
        range_size = 1000
        self.batch_size = 90
        epoch_size = 100
        self.image_size = (140, 140)
        self.train_image_list, self.train_label_list = get_image_paths_and_labels(train_set)

        self.index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                                                    shuffle=True, seed=None, capacity=32)

        self.index_dequeue_op = self.index_queue.dequeue_many(self.batch_size * epoch_size, 'index_dequeue')

        self.learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        self.batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        self.phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        self.image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='labels')
        self.control_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='control')

        nrof_preprocess_threads = 4
        self.input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                              dtypes=[tf.string, tf.int32, tf.int32],
                                              shapes=[(1,), (1,), (1,)],
                                              shared_name=None, name=None)
        self.enqueue_op = self.input_queue.enqueue_many(
            [self.image_paths_placeholder, self.labels_placeholder, self.control_placeholder],
            name='enqueue_op')
        self.image_batch, self.label_batch = create_input_pipeline(self.input_queue, self.image_size,
                                                                   nrof_preprocess_threads, self.batch_size_placeholder)

    def train(self):
        learning_rate_schedule_file = '/home/xiajun/dev/facerec/facenet/data/learning_rate_schedule_classifier_vggface2.txt'
        args_learning_rate = -1
        random_rotate = True
        random_crop = True
        random_flip = True
        prelogits_hist_max = 10.0
        use_fixed_image_standardization = True

        lr = 0.05

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            print(self.index_queue.size().eval())
            index_epoch = sess.run(self.index_dequeue_op)
            print(self.index_queue.size().eval())
            label_epoch = np.array(self.train_label_list)[index_epoch]
            image_epoch = np.array(self.train_image_list)[index_epoch]

            # Enqueue one epoch of image paths and labels
            labels_array = np.expand_dims(np.array(label_epoch), 1)
            image_paths_array = np.expand_dims(np.array(image_epoch), 1)
            control_value = RANDOM_ROTATE * random_rotate + \
                            RANDOM_CROP * random_crop + \
                            RANDOM_FLIP * random_flip + \
                            FIXED_STANDARDIZATION * use_fixed_image_standardization
            control_array = np.ones_like(labels_array) * control_value

            print(self.input_queue.size().eval())
            sess.run(self.enqueue_op,
                     {self.image_paths_placeholder: image_paths_array, self.labels_placeholder: labels_array,
                      self.control_placeholder: control_array})
            print(self.input_queue.size().eval())

            # self.debug_tmp(sess)

            feed_dict = {self.batch_size_placeholder: self.batch_size}
            count = 0
            while True:
                arr = sess.run(self.image_batch, feed_dict=feed_dict)
                print(count, arr.shape, arr.mean(), arr.std())
                count += 1
            print('debug')

    def debug_tmp(self, sess):
        example = self.input_queue.dequeue()
        filenames, label, control = sess.run(example)
        for filename in tf.unstack(filenames):
            file_contents = tf.read_file(filename)
            image = tf.image.decode_image(file_contents, 3)
            src_image = sess.run(image)
            misc.imsave('src.jpg', src_image)
            image = tf.cond(get_control_flag(control[0], RANDOM_ROTATE),
                            lambda: tf.py_func(random_rotate_image, [image], tf.uint8),
                            lambda: tf.identity(image))
            rotate_image = sess.run(image)
            misc.imsave('rotate_image.jpg', rotate_image)

            image = tf.cond(get_control_flag(control[0], RANDOM_CROP),
                            lambda: tf.random_crop(image, self.image_size + (3,)),
                            lambda: tf.image.resize_image_with_crop_or_pad(image, self.image_size[0],
                                                                           self.image_size[1]))
            crop_image = sess.run(image)
            misc.imsave('crop_image.jpg', crop_image)

            image = tf.cond(get_control_flag(control[0], RANDOM_FLIP),
                            lambda: tf.image.random_flip_left_right(image),
                            lambda: tf.identity(image))
            flip_image = sess.run(image)
            misc.imsave('flip_image.jpg', flip_image)

            f32image = tf.cast(image, tf.float32)
            image = sess.run(image)
            f32image = sess.run(f32image)

            image = tf.cond(get_control_flag(control[0], FIXED_STANDARDIZATION),
                            lambda: (tf.cast(image, tf.float32) - 127.5) / 128.0,
                            lambda: tf.image.per_image_standardization(image))
            fixstd_image = sess.run(image)
            misc.imsave('fixstd_image.jpg', fixstd_image)

            image = tf.cond(get_control_flag(control[0], FLIP),
                            lambda: tf.image.flip_left_right(image),
                            lambda: tf.identity(image))


if __name__ == '__main__':
    data_dir = '/home/xiajun/dataset/gc_together/origin_gen90_align160_margin32'
    train_set, valid_set = load_data(data_dir)

    queue = QueueTrain(train_set)
    queue.train()
    print('debug')
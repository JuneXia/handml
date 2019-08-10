import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import os

# tf.enable_eager_execution()

# tf.executing_eagerly()

DEBUG = True


# 数据集结构
def demo1():
    dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
    print(dataset1.output_types)  # ==> "tf.float32"
    print(dataset1.output_shapes)  # ==> "(10,)"

    dataset2 = tf.data.Dataset.from_tensor_slices(
        (tf.random_uniform([4]),
         tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
    print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
    print(dataset2.output_shapes)  # ==> "((), (100,))"

    dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
    print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
    print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"

    dataset1 = dataset1.map(lambda x: ...)

    dataset2 = dataset2.flat_map(lambda x, y: ...)

    # Note: Argument destructuring is not available in Python 3.
     #dataset3 = dataset3.filter(lambda x, (y, z): ...)


# 创建迭代器
## 单次迭代器
def demo2():
    dataset = tf.data.Dataset.range(100)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        for i in range(100):
            value = sess.run(next_element)

            assert i == value

## 可初始化迭代器
def demo3():
    max_value = tf.placeholder(tf.int64, shape=[])
    dataset = tf.data.Dataset.range(max_value)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        # Initialize an iterator over a dataset with 10 elements.
        sess.run(iterator.initializer, feed_dict={max_value: 10})
        for i in range(10):
            value = sess.run(next_element)
            assert i == value

        # Initialize the same iterator over a dataset with 100 elements.
        sess.run(iterator.initializer, feed_dict={max_value: 100})
        for i in range(100):
            value = sess.run(next_element)
            assert i == value


## 可重新初始化迭代器可以通过多个不同的 Dataset 对象进行初始化
def demo4():
    # Define training and validation datasets with the same structure.
    training_dataset = tf.data.Dataset.range(100).map(
        lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
    validation_dataset = tf.data.Dataset.range(50)

    # A reinitializable iterator is defined by its structure. We could use the
    # `output_types` and `output_shapes` properties of either `training_dataset`
    # or `validation_dataset` here, because they are compatible.
    iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                               training_dataset.output_shapes)
    next_element = iterator.get_next()

    training_init_op = iterator.make_initializer(training_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)

    with tf.Session() as sess:
        # Run 20 epochs in which the training dataset is traversed, followed by the
        # validation dataset.
        for _ in range(20):
            # Initialize an iterator over the training dataset.
            sess.run(training_init_op)
            for _ in range(100):
                sess.run(next_element)

            # Initialize an iterator over the validation dataset.
            sess.run(validation_init_op)
            for _ in range(50):
                sess.run(next_element)


## 可馈送迭代器可以与 tf.placeholder 一起使用
def demo5():
    # Define training and validation datasets with the same structure.
    training_dataset = tf.data.Dataset.range(100).map(
        lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
    validation_dataset = tf.data.Dataset.range(50)

    # A feedable iterator is defined by a handle placeholder and its structure. We
    # could use the `output_types` and `output_shapes` properties of either
    # `training_dataset` or `validation_dataset` here, because they have
    # identical structure.
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, training_dataset.output_types, training_dataset.output_shapes)
    next_element = iterator.get_next()

    # You can use feedable iterators with a variety of different kinds of iterator
    # (such as one-shot and initializable iterators).
    training_iterator = training_dataset.make_one_shot_iterator()
    validation_iterator = validation_dataset.make_initializable_iterator()

    with tf.Session() as sess:
        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())

        # Loop forever, alternating between training and validation.
        while True:
            # Run 200 steps using the training dataset. Note that the training dataset is
            # infinite, and we resume from where we left off in the previous `while` loop
            # iteration.
            for _ in range(200):
                sess.run(next_element, feed_dict={handle: training_handle})

            # Run one pass over the validation dataset.
            sess.run(validation_iterator.initializer)
            for _ in range(50):
                sess.run(next_element, feed_dict={handle: validation_handle})


# 消耗迭代器中的值
def demo6():
    dataset = tf.data.Dataset.range(5)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Typically `result` will be the output of a model, or an optimizer's
    # training operation.
    result = tf.add(next_element, next_element)

    with tf.Session() as sess:
        """
        sess.run(iterator.initializer)
        print(sess.run(result))  # ==> "0"
        print(sess.run(result))  # ==> "2"
        print(sess.run(result))  # ==> "4"
        print(sess.run(result))  # ==> "6"
        print(sess.run(result))  # ==> "8"
        try:
            sess.run(result)
        except tf.errors.OutOfRangeError:
            print("End of dataset")  # ==> "End of dataset"
        """

        sess.run(iterator.initializer)
        while True:
            try:
                sess.run(result)
            except tf.errors.OutOfRangeError:
                break

# 使用 Dataset.map() 预处理数据
## 解析 tf.Example 协议缓冲区消息
def demo7():
    dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
    dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))
    dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

    iterator = dataset3.make_initializable_iterator()
    next1, (next2, next3) = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)

        print(sess.run(next1))
        print(sess.run(next2))
        print(sess.run(next3))

    # Create saveable object from iterator.
    # saveable = tf.contrib.data.make_saveable_from_iterator(iterator)
    saveable = tf.data.experimental.make_saveable_from_iterator(iterator)

    # Save the iterator state by adding it to the saveable objects collection.
    tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
    saver = tf.train.Saver()

    ckpt_save_path = './save_model'
    with tf.Session() as sess:
        saver.save(sess, ckpt_save_path)

    # Restore the iterator state.
    with tf.Session() as sess:
        saver.restore(sess, ckpt_save_path)


def get_dataset(data_path):
    images_path = []
    images_label = []
    images = os.listdir(data_path)
    images.sort()
    if DEBUG:
        images = images[0:2]
    for i, image in enumerate(images):
        imgs = os.listdir(os.path.join(data_path, image))
        if DEBUG:
            imgs = imgs[0:1000]
        images_path.extend([os.path.join(data_path, image, img) for img in imgs])
        images_label.extend([i] * len(imgs))

    return images_path, images_label


## 解码图片数据并调整其大小，批处理，随机重排
def demo8():
    # Reads an image from a file, decodes it into a dense tensor, and resizes it
    # to a fixed shape.
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [28, 28])
        return image_resized, label

    data_path = '/home/xiaj/res/mnist'
    images_path, images_label = get_dataset(data_path)

    # A vector of filenames.
    filenames = tf.constant(images_path)

    # `labels[i]` is the label for the image in `filenames[i].
    labels = tf.constant(images_label)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=2000, seed=666, reshuffle_each_iteration=True)
    dataset = dataset.repeat(5)
    dataset = dataset.batch(100)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        while True:
            try:
                images, labels = sess.run(next_element)
                print('images.shape={}, labels.shape={}, labels={}'.format(images.shape, labels.shape, labels))
            except tf.errors.OutOfRangeError as e:
                # 迭代完后，如果还想要继续从头迭代，可以再次sess.run(iterator.initializer)即可。
                print(e)


## 使用 tf.py_func() 应用任意 Python 逻辑
def demo9():
    import cv2

    # Use a custom OpenCV function to read the image, instead of the standard
    # TensorFlow `tf.read_file()` operation.
    def _read_py_function(filename, label):
        # image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
        image_decoded = cv2.imread(filename.numpy().decode(), cv2.IMREAD_GRAYSCALE)
        image_decoded = image_decoded.reshape((28, 28, 1))
        return image_decoded, label

    # Use standard TensorFlow operations to resize the image to a fixed shape.
    def _resize_function(image_decoded, label):
        image_decoded.set_shape([None, None, None])
        image_resized = tf.image.resize_images(image_decoded, [32, 32])
        return image_resized, label

    data_path = '/home/xiaj/res/mnist'
    images_path, images_label = get_dataset(data_path)

    dataset = tf.data.Dataset.from_tensor_slices((images_path, images_label))
    dataset = dataset.map(
        lambda filename, label: tuple(tf.py_function(
            _read_py_function, [filename, label], [tf.uint8, label.dtype])))
    dataset = dataset.map(_resize_function)

    dataset = dataset.shuffle(buffer_size=2000, seed=666, reshuffle_each_iteration=True)
    dataset = dataset.repeat(5)
    dataset = dataset.batch(100)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        while True:
            try:
                images, labels = sess.run(next_element)
                print('images.shape={}, labels.shape={}, labels={}'.format(images.shape, labels.shape, labels))
            except tf.errors.OutOfRangeError as e:
                # 迭代完后，如果还想要继续从头迭代，可以再次sess.run(iterator.initializer)即可。
                print(e)


if __name__ == '__main__':
    demo9()

    print('debug')


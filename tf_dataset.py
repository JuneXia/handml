import tensorflow as tf
import numpy as np


# ********************************************************
dataset = tf.data.Dataset.from_tensor_slices(np.array([1,2,3,4,5]))
dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=(5, 2)))
dataset = tf.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1,2,3,4,5]),
        "b": np.random.uniform(size=(5, 2))
    }
)
dataset = tf.data.Dataset.from_tensor_slices(
    (np.array([1,2,3,4,5]), np.random.uniform(size=(5, 2)))
)


def transform_image():
    # 函数的功能时将filename对应的图片文件读进来，并缩放到统一的大小
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        image_tensor = tf.convert_to_tensor(image_decoded)
        image_tensor.set_shape([12, 12, 3])
        # image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, 250, 250)  # ok
        image_resized = tf.image.resize_images(image_tensor, [28, 28])
        return image_resized, label

    # 图片文件的列表
    filenames = tf.constant([
        '/home/xiajun/res/face/VGGFace2/Experiment/train_mtcnn_align160x160_margin32/n000124/0001_01.jpg',
        '/home/xiajun/res/face/VGGFace2/Experiment/train_mtcnn_align160x160_margin32/n000124/0002_01.jpg',
        '/home/xiajun/res/face/VGGFace2/Experiment/train_mtcnn_align160x160_margin32/n000124/0003_01.jpg',
        '/home/xiajun/res/face/VGGFace2/Experiment/train_mtcnn_align160x160_margin32/n000124/0004_01.jpg',
        '/home/xiajun/res/face/VGGFace2/Experiment/train_mtcnn_align160x160_margin32/n000124/0005_01.jpg',
        '/home/xiajun/res/face/VGGFace2/Experiment/train_mtcnn_align160x160_margin32/n000124/0006_01.jpg'
    ])
    # label[i]就是图片filenames[i]的label
    labels = tf.constant([0, 1, 2, 3, 4, 5])

    # 此时dataset中的一个元素是(filename, label)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # 此时dataset中的一个元素是(image_resized, label)
    dataset = dataset.map(_parse_function)

    # 此时dataset中的一个元素是(image_resized_batch, label_batch)
    # dataset = dataset.shuffle(buffer_size=1000).batch(32).repeat(10)
    dataset = dataset.shuffle(buffer_size=1000).batch(2).repeat(5)

    return dataset

dataset = transform_image()

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
# ********************************************************



# ********************************************************
# initializable iterator
limit = tf.placeholder(shape=[], dtype=tf.int32)
dataset = tf.data.Dataset.from_tensor_slices(tf.range(start=0, limit=limit))
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={limit: 10})
    for i in range(10):
      value = sess.run(next_element)
      assert i == value
# ********************************************************




# ********************************************************
# 从硬盘中读入两个Numpy数组
with np.load("/var/data/training_data.npy") as data:
  features = data["features"]
  labels = data["labels"]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
iterator = dataset.make_initializable_iterator()
sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})
# ********************************************************

print('debug')

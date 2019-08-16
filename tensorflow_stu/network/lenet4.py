import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


# Change Mark: 相对于lenet2.py，本次实验使用eager模式时，在第一次调用call函数时，
# 传入的inputs是整个训练集（亲测第二次及以后都是每次只传入batch_size个训练集），这将导致内存溢出。
# 但本次实验不使用eager模式是ok的。如果非要想使用eager模式，可以试试将训练用tf.data封装后再传入model.fit。
# tf.enable_eager_execution()
print('is eager executing: ', tf.executing_eagerly())


class LeNet(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.num_classes = num_classes

        self.conv1_fn = layers.Conv2D(6, (5, 5), activation='relu', padding='same', name='conv1')
        self.pool1_fn = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool1')
        self.conv2_fn = layers.Conv2D(16, (5, 5), activation='relu', padding='valid', name='conv2')
        self.pool2_fn = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')
        self.conv3_fn = layers.Conv2D(120, [5, 5], activation='relu', padding='valid', name='conv3')
        self.flat_fn = layers.Flatten(name='flatten')
        self.dense1_fn = layers.Dense(units=84, activation='relu', name='fc2')

    # python3应该是__call__吧
    def call(self, inputs):
        x = self.conv1_fn(inputs)
        x = self.pool1_fn(x)
        x = self.conv2_fn(x)
        x = self.pool2_fn(x)
        x = self.conv3_fn(x)
        x = self.flat_fn(x)
        x = self.dense1_fn(x)

        return x

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape = [shape[0], 84]
        return tf.TensorShape(shape)


if __name__ == '__main__':
    (mnist_images, mnist_labels), a = tf.keras.datasets.mnist.load_data()
    mnist_labels = to_categorical(mnist_labels, 10)
    mnist_images = np.expand_dims(mnist_images, axis=3) / 255
    mnist_labels = mnist_labels.astype(np.float)

    model = tf.keras.Sequential([
        LeNet(),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(mnist_images, mnist_labels, batch_size=32, epochs=5)

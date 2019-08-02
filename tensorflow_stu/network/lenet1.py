import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


if __name__ == '__main__':
    (mnist_images, mnist_labels), a = tf.keras.datasets.mnist.load_data()
    mnist_labels = to_categorical(mnist_labels, 10)
    mnist_images = np.expand_dims(mnist_images, axis=3) / 255
    mnist_labels = mnist_labels.astype(np.float)

    model = tf.keras.Sequential([
        layers.Conv2D(6, (5, 5), activation='relu', padding='same', name='conv1'),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool1'),
        layers.Conv2D(16, (5, 5), activation='relu', padding='valid', name='conv2'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2'),
        layers.Conv2D(120, [5, 5], activation='relu', padding='valid', name='conv3'),
        layers.Flatten(name='flatten'),
        layers.Dense(units=84, activation='relu', name='fc2'),

        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(mnist_images, mnist_labels, batch_size=32, epochs=5)


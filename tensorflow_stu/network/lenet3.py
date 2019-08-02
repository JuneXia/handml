import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


# Change Mark: 相对于lenet2.py，本次实验必须使用eager模式，否则会报错
tf.enable_eager_execution()
print('is eager executing: ', tf.executing_eagerly())


class LeNet(keras.Model):
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        # super(LeNet, self).__init__(name="LeNet")
        self.num_classes = num_classes

        img_input = layers.Input(shape=input_shape)

        x = layers.Conv2D(6, (5, 5), activation='relu', padding='same', name='conv1')(img_input)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
        x = layers.Conv2D(16, (5, 5), activation='relu', padding='valid', name='conv2')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(x)
        x = layers.Conv2D(120, [5, 5], activation='relu', padding='valid', name='conv3')(x)
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(units=84, activation='relu', name='fc2')(x)

        # 调用Model类的Model(input, output, name="***")构造方法
        super(LeNet, self).__init__(img_input, x, name="LeNet")

    def __call__(self, *args, **kwargs):
        '''
        TODO: 实测该函数确实会被调用，但该函数在子类是否实现并无影响。
        '''
        print('debug')

    def call(self, inputs):
        '''
        TODO: 实测该函数并不会被调用
        '''
        # 前向传播计算
        # 使用在__init__方法中定义的层
        return self.output(inputs)


if __name__ == '__main__':
    (mnist_images, mnist_labels), a = tf.keras.datasets.mnist.load_data()
    mnist_labels = to_categorical(mnist_labels, 10)
    mnist_images = np.expand_dims(mnist_images, axis=3) / 255
    mnist_labels = mnist_labels.astype(np.float)

    # Change Mark: 相对于lenet2.py，本次实验中的LeNet只是整个网络中的一部分，使用tf.keras.Sequential对网络进行拼接。
    model = tf.keras.Sequential([
        LeNet(),

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

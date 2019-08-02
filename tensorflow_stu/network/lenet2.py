import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


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

        x = layers.Dense(units=num_classes, activation='softmax', name='prediction')(x)

        # 调用Model类的Model(input, output, name="***")构造方法
        super(LeNet, self).__init__(img_input, x, name='LeNet')
        # 本次实验是使用tf.keras的函数式API编程，上面这种调用父类构造的方法和下面的直接创建父类对象效果是一样的。
        # tf.keras.Model(inputs=img_input, outputs=x, name='LeNet')
        # 如果要用tf.keras.Model创建父类对象的话，那么子类LeNet也没有必要继承keras.Model了。


    def __call__(self, *args, **kwargs):
        '''
        TODO: 实实测该函数并不会被调用
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

    model = LeNet()

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(mnist_images, mnist_labels, batch_size=32, epochs=5)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.contrib.slim as slim
from tensorflow.keras.utils import to_categorical


def inference(inputs, keep_probability, phase_train=True,
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return lenet(inputs, is_training=phase_train, dropout_keep_prob=keep_probability,
                     bottleneck_layer_size=bottleneck_layer_size, reuse=reuse)


def lenet(inputs, is_training=True,
          dropout_keep_prob=0.8,
          bottleneck_layer_size=128,
          reuse=None,
          scope='InceptionResnetV1'):
    end_points = {}

    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
                net = end_points['conv1'] = slim.conv2d(inputs, 32, [5, 5], stride=1, scope='conv1')
                net = end_points['pool1'] = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')
                net = end_points['conv2'] = slim.conv2d(net, 64, [5, 5], stride=1, scope='conv2')
                net = end_points['pool2'] = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')
                net = slim.flatten(net)
                end_points['Flatten'] = net

                net = end_points['fc3'] = slim.fully_connected(net, 1024, scope='fc3')

                net = end_points['dropout3'] = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                                            scope='dropout3')
                # logits = end_points['Logits'] = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc4')

                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, scope='Bottleneck',
                                           reuse=False)

    return net, end_points


class LeNet(keras.Model):
    def __init__(self, num_classes=10):
        self.output_dim = num_classes
        super(LeNet, self).__init__()

        img_input = layers.Input(shape=(28, 28, 1))
        # img_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

        x, _ = inference(img_input, 0.5, phase_train=True, bottleneck_layer_size=128,
                                  weight_decay=0.0, reuse=None)

        x = layers.Dense(units=num_classes, activation='softmax', name='prediction')(x)

        # 调用Model类的Model(input, output, name="***")构造方法
        super(LeNet, self).__init__(img_input, x, name='LeNet')

    def call(self, inputs, training=None, mask=None):
        self.outputs(inputs)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(LeNet, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


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

import os
from keras import backend as K
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
tf.compat.v1.enable_eager_execution()
print('is eager execution: ', tf.compat.v1.executing_eagerly())

if __name__ == '__main__1':  # 类的继承实验
    class Base(object):
        def __init__(self):
            self.num = 0

        def __call__(self, *args, **kwargs):
            print(args)


    class Human(Base):
        def __init__(self):
            super(Human, self).__init__()
            print(self.num)

        def __call__(self, *args, **kwargs):
            print(args)

    person = Human()
    person()

    print('debug')


if __name__ == '__main__2':  # tf1.x的tf.data在每个epoch后不会自动reshuffle.
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, optimizers, datasets

    print('tf.__version__: ', tf.__version__)
    tf.compat.v1.enable_eager_execution()
    print('is eager executing: ', tf.compat.v1.executing_eagerly())

    (x, y), (x_val, y_val) = datasets.mnist.load_data()
    x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    # y = tf.one_hot(y, depth=10)
    print(x.shape, y.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    train_dataset = train_dataset.shuffle(1000, reshuffle_each_iteration=True).batch(100)
    epoch_size = 5
    for epoch in range(epoch_size):
        print('\nEpoch {}/{}:'.format(epoch, epoch_size))

        for step, (images, labels) in enumerate(train_dataset):
            if step < 2:
                print(labels.numpy())

        continue

        train_iter = iter(train_dataset)
        step = 0
        while True:
            try:
                images, labels = next(train_iter)
                if step < 2:
                    print(labels.numpy())
            except StopIteration:
                break
            step += 1


if __name__ == '__main__':  # 范数实验
    arr = np.array([[0,1,2],
                   [2,0,3]], dtype=np.float32)
    a = tf.norm(arr)
    a = tf.nn.l2_normalize(arr, axis=1)
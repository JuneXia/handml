import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


# Change Mark: 相对于lenet2.py，本次实验使用eager模式时，在第一次调用call函数时，
# 传入的inputs是整个训练集（亲测第二次及以后都是每次只传入batch_size个训练集），这将导致内存溢出。
# 但本次实验不使用eager模式是ok的。如果非要想使用eager模式，可以试试将训练用tf.data封装后再传入model.fit。
tf.enable_eager_execution()
print('is eager executing: ', tf.executing_eagerly())


class SiameseNet(tf.keras.Model):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def call(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape = [shape[0], 84]
        return tf.TensorShape(shape)

    def get_embedding(self, x):
        return self.embedding_net(x)



class ContrastiveLoss(tf.keras.Model):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def call(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

# -*- coding: UTF-8 -*-
import tensorflow as tf


class ContrastiveLoss(tf.keras.layers.Layer):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def __call__(self, *args, **kwargs):
        labels, (pred1, pred2) = args

        eps = 1e-9
        margin = 1.
        size_average = True
        square = tf.pow(pred2 - pred1, 2)
        distances = tf.reduce_sum(square, axis=1)

        tmp2 = tf.keras.activations.relu(margin - tf.sqrt(distances + eps))
        tmp2 = tf.pow(tmp2, 2)
        tmp1 = labels * distances + (1 + -1 * labels) * tmp2
        losses = 0.5 * tmp1
        if size_average:
            loss_step = tf.reduce_mean(losses)
        else:
            loss_step = tf.reduce_sum(losses)

        return loss_step


class ComplexLoss(tf.keras.layers.Layer):
    def __init__(self):
        super(ComplexLoss, self).__init__()
        self.bincross_loss_func = tf.keras.losses.BinaryCrossentropy()
        self.contrast_loss_func = ContrastiveLoss()

    def __call__(self, *args, **kwargs):
        labels, (outputs, (emb1, emb2)) = args
        bincross_loss = self.bincross_loss_func(labels, outputs)
        contrast_loss = self.contrast_loss_func(labels, (emb1, emb2))

        # ok
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.add_loss(bincross_loss, inputs=True)
        self.add_loss(contrast_loss, inputs=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # loss = tf.reduce_mean((bincross_loss, contrast_loss))
        # self.add_loss(loss, inputs=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        return self.losses

    def call_need_tuple_when_invoke_function(self, inputs, **kwargs):  # 调用函数时需要传元祖
        labels, outputs, (emb1, emb2) = inputs
        bincross_loss = self.bincross_loss_func(labels, outputs)
        contrast_loss = self.contrast_loss_func(labels, (emb1, emb2))
        self.add_loss(bincross_loss, inputs=True)
        self.add_loss(contrast_loss, inputs=True)

        # self.add_metric(bincross_loss)

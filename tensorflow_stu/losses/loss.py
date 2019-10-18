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


from tensorflow.python.keras import losses


class PrelogitsNormLoss(losses.Loss):
    def __init__(self, reduction="sum_over_batch_size", name=None):
        super(PrelogitsNormLoss, self).__init__(reduction=reduction, name=name)

        self.eps = 1e-4
        self.prelogits_norm_p = 1
        self.prelogits_norm_loss_factor = 5e-4
        # prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(model.output) + eps, ord=prelogits_norm_p, axis=1))
        # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * prelogits_norm_loss_factor)

    def call(self, y_true, y_pred):
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(y_pred) + self.eps, ord=self.prelogits_norm_p, axis=1))
        return prelogits_norm * self.prelogits_norm_loss_factor


class ComplexLoss2(losses.Loss):
    def __init__(self):
        raise Exception('写的不对！')
        super(ComplexLoss2, self).__init__()
        self.cross_loss_func = tf.keras.losses.CategoricalCrossentropy()
        self.prelogits_loss_func = PrelogitsNormLoss()

    def __call__(self, *args, **kwargs):
        labels, outputs = args
        cross_loss = self.cross_loss_func(labels, outputs)
        logits_loss = self.prelogits_loss_func(labels, outputs)

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


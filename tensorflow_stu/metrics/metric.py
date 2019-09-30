# -*- coding: UTF-8 -*-
import tensorflow as tf
from utils import tools
from utils import evaluate
import numpy as np


class AccumulatedLossMetric(tf.keras.layers.Layer):
    def __init__(self, name='Loss'):
        super(AccumulatedLossMetric, self).__init__()
        self.metric_loss = tf.keras.metrics.Mean(name)

    def __call__(self, *args, **kwargs):
        labels, predicts, loss_step = args
        return self.metric_loss(loss_step)

    def name(self):
        return self.metric_loss.name

    def result(self):
        return self.metric_loss.result()

    def reset_states(self):
        self.metric_loss.reset_states()


class AccumulatedAccuaracyMetric(tf.keras.layers.Layer):
    def __init__(self, name='acc'):
        super(AccumulatedAccuaracyMetric, self).__init__()
        self.metric_acc = tf.keras.metrics.Accuracy(name)

    def __call__(self, *args, **kwargs):
        labels, predicts, loss_step = args
        return self.metric_acc(labels, predicts)

    def name(self):
        return self.metric_acc.name

    def result(self):
        return self.metric_acc.result()

    def reset_states(self):
        self.metric_acc.reset_states()


class AccumulatedBinaryAccuracyMetric(tf.keras.layers.Layer):
    def __init__(self, name='acc'):
        super(AccumulatedBinaryAccuracyMetric, self).__init__()
        self.metric_acc = tf.keras.metrics.BinaryAccuracy(name)

    def __call__(self, *args, **kwargs):
        labels, (predicts, _), loss_step = args
        return self.metric_acc(labels, predicts)

    def name(self):
        return self.metric_acc.name

    def result(self):
        return self.metric_acc.result()

    def reset_states(self):
        self.metric_acc.reset_states()


class AccumulatedEmbeddingAccuracyMetric(tf.keras.layers.Layer):
    def __init__(self, name):
        super(AccumulatedEmbeddingAccuracyMetric, self).__init__()
        self.name = name
        self.distance_metric = 0

    def __call__(self, *args, **kwargs):
        labels, (embeddings1, embeddings2), loss_step = args
        if type(embeddings1) != np.ndarray:
            embeddings1 = embeddings1.numpy()
        if type(embeddings2) != np.ndarray:
            embeddings2 = embeddings2.numpy()

        tpr, fpr, thresholds_roc, accuracy, val, val_std, far = evaluate.evaluate(embeddings1, embeddings2, labels,
                                                                                  nrof_folds=10,
                                                                                  distance_metric=self.distance_metric,
                                                                                  subtract_mean=True)
        auc = tools.compute_auc(fpr, tpr)

        '''
        val = '%.4f±%.3f' % (val, val_std)
        acc = '%.4f±%.3f' % (accuracy.mean(), accuracy.std())
        auc = '%.4f' % auc
        far = '%.4f' % far

        print('acc:{}, auc={}, val:{}@far:{}'.format(acc, auc, val, far))
        '''

        return accuracy.mean(), accuracy.std(), auc, val, val_std, far

    def name(self):
        return self.name

    def result(self):
        return 1

    def reset_states(self):
        pass

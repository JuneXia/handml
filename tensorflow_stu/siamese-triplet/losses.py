import tensorflow as tf


class ContrastiveLoss(object):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def loss(self, input1, input2, target, size_average=True):
        return 1.0


class RegularizationLoss(object):
    def __init__(self, prelogits):
        super(RegularizationLoss, self).__init__()

        prelogits_norm_p = 1.0
        prelogits_norm_loss_factor = 0.9
        eps = 1e-4
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=prelogits_norm_p, axis=1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * prelogits_norm_loss_factor)

        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64)),
                                     tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        self.train_op = facenet.train(total_loss, global_step, optimizer,
                                      self.learning_rate, moving_average_decay, tf.global_variables(),
                                      log_histograms)
        """


class ClassificationLoss(object):
    def __init__(self, labels, logits):
        super(ClassificationLoss, self).__init__()

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection(tf.GraphKeys.LOSSES, cross_entropy_mean)

        """
        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64)),
                                     tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        self.train_op = facenet.train(total_loss, self.global_step, optimizer,
                                      self.learning_rate, moving_average_decay, tf.global_variables(),
                                      log_histograms)
        """

class TotalLoss(object):
    def __init__(self):
        super(TotalLoss, self).__init__()

        losses = tf.get_collection(tf.GraphKeys.LOSSES)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.total_loss = tf.add_n(losses + regularization_losses, name='total_loss')


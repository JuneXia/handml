import tensorflow as tf
from visualize import mnist_inference
import os

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99

LOG_DIR = './visualize/log'
SPRITE_FILE = 'mnist_sprite.jpg'
META_FIEL = "mnist_meta.tsv"
TENSOR_NAME = "FINAL_LOGITS"


class DummyFileWriter(object):
  def get_logdir(self):
    return './visualize/log'


def visualisation(final_result):
    y = tf.Variable(final_result, name=TENSOR_NAME)
    # summary_writer = tf.summary.FileWriter(LOG_DIR)
    summary_writer = tf.contrib.summary.create_file_writer(LOG_DIR, flush_millis=1000)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = y.name

    # Specify where you find the metadata
    embedding.metadata_path = META_FIEL

    # Specify where you find the sprite (we will create this later)
    embedding.sprite.image_path = SPRITE_FILE
    embedding.sprite.single_image_dim.extend([28, 28])

    # Say that you want to visualise the embeddings
    # projector.visualize_embeddings(summary_writer, config)
    projector.visualize_embeddings(DummyFileWriter(), config)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(LOG_DIR, "model"), TRAINING_STEPS)

    summary_writer.close()


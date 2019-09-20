import os
import numpy as np
import tensorflow as tf
import cv2

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    save_path = '/home/xiaj/res/mnist'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    images = np.concatenate((x_train, x_test))
    labels = np.concatenate((y_train, y_test))

    for i, (image, label) in enumerate(zip(images, labels)):
        image_path = os.path.join(save_path, str(label))
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        filename = os.path.join(image_path, '%(label)05d.jpg' % {'label': i})
        cv2.imwrite(filename, image)

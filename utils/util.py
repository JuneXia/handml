# -*- coding: UTF-8 -*-
import os
import numpy as np


DEBUG = True

def get_dataset(data_path, shuffle=True, validation_ratio=0.0):
    '''
data_path dir style:
data
├── folder1
│   ├── 00063.jpg
│   ├── 00068.jpg
└── folder2
    ├── 00070.jpg
    ├── 00072.jpg

    :param data_path:
    :return:
    '''
    images_path = []
    images_label = []
    images = os.listdir(data_path)
    images.sort()
    if DEBUG:
        images = images[0:2]
    for i, image in enumerate(images):
        imgs = os.listdir(os.path.join(data_path, image))
        images_path.extend([os.path.join(data_path, image, img) for img in imgs])
        images_label.extend([i] * len(imgs))

    if shuffle:
        images = np.array([images_path, images_label]).transpose()
        np.random.shuffle(images)
        images_path = images[:, 0]
        images_label = images[:, 1].astype(np.int32)

    if DEBUG:
        images_path = images_path[0:500]
        images_label = images_label[0:500]

    if validation_ratio > 0.0:
        if not shuffle:
            raise Exception('When there is a validation set split requirement, shuffle must be True.')
        validation_size = int(len(images_path) * validation_ratio)
        validation_images_path = images_path[0:validation_size]
        validation_images_label = images_label[0:validation_size]

        train_images_path = images_path[validation_size:]
        train_images_label = images_label[validation_size:]
        return train_images_path, train_images_label, validation_images_path, validation_images_label
    else:
        return images_path, images_label
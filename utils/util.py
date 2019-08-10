import os
import numpy as np


DEBUG = True

def get_dataset(data_path, shuffle=True):
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
        if DEBUG:
            imgs = imgs[0:1000]
        images_path.extend([os.path.join(data_path, image, img) for img in imgs])
        images_label.extend([i] * len(imgs))

    if shuffle:
        images = np.array([images_path, images_label]).transpose()
        np.random.shuffle(images)
        images_path = images[:, 0]
        images_label = images[:, 1].astype(np.int32)

    return images_path, images_label
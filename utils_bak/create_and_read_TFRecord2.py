import tensorflow as tf
import numpy as np
import os
img_width = 224
img_height = 224


def get_file(file_dir):
    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        for name in files:
            images.append(os.path.join(root, name))
        for name in sub_folders:
            temp.append(os.path.join(root, name))
    labels = []
    for one_folder in temp:
        n_img = len(os.listdir(one_folder))
        letter = one_folder.split('\\')[-1]
        if letter == 'cat':
            labels = np.append(labels, n_img * [0])
        else:
            labels = np.append(labels, n_img * [1])
    # shuffle
    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]

    return image_list, label_list



def get_batch(image_list, label_list,img_width,img_height,batch_size,capacity):
    image = tf.cast(image_list,tf.string)
    label = tf.cast(label_list,tf.int32)

    input_queue = tf.train.slice_input_producer([image,label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image,img_width,img_height)
    # image = tf.image.per_image_standardization(image) #将图片标准化
    image_batch,label_batch = tf.train.batch([image,label],batch_size=batch_size,num_threads=64,capacity=capacity)
    label_batch = tf.reshape(label_batch,[batch_size])

    return image_batch,label_batch


imdir = '/home/xiajun/dev/alg_verify/FasterRCNN_Instrument/images'
namelist = os.listdir(imdir)
image_list = []
label_list = []
for name in namelist:
    imagefile = os.path.join(imdir, name)
    image_list.append(imagefile)
    label_list.append(0)


image_batch, label_batch = get_batch(image_list, label_list, img_width, img_height, 25, 50)
x_imgs = tf.placeholder(tf.float32, [None, img_width, img_height, 3])
y_imgs = tf.placeholder(tf.int32, [None, 2])


sess = tf.Session()
sess.run(tf.global_variables_initializer())
# vgg.load_weights('../vgg16_weights_and_classes/vgg16_weights.npz',sess)
# saver = vgg.saver()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord,sess=sess)

import time
import cv2
start_time = time.time()

for i in range(100):
    image, label = sess.run([image_batch, label_batch])
    count = 0
    for im in image:
        cv2.imwrite(str(count) + '.jpg', im)
        count += 1
    # labels = onehot(label)

    # sess.run(optimizer, feed_dict={x_imgs: image, y_imgs: labels})
    # loss_record = sess.run(loss, feed_dict={x_imgs: image, y_imgs: labels})
    print("now the loss is %f " % loss_record)
    end_time = time.time()
    print('time: ', (end_time - start_time))
    start_time = end_time
    print("----------epoch %d is finished---------------" % i)

saver.save(sess,".\\vgg_finetuning_model\\")
print("Optimization Finished!")
coord.request_stop()
coord.join(threads)




print('debug')
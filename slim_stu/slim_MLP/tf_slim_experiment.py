"""
# 程序19-1： 模型变量
import tensorflow.contrib.slim as slim
import tensorflow as tf

weight1 = slim.model_variable('weight1',
                            shape=[2, 3],
                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                            regularizer=slim.l2_regularizer(0.05))

weight2 = slim.model_variable('weight2',
                            shape=[2, 3],
                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                            regularizer=slim.l2_regularizer(0.05))

model_variables = slim.get_model_variables()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(weight1))
    print("--------------------")
    print(sess.run(model_variables))
    print("--------------------")
    print(sess.run(slim.get_variables_by_suffix("weight1")))
"""


"""
# 程序19-2：普通变量
import tensorflow.contrib.slim as slim
import tensorflow as tf

weight1 = slim.variable('weight1',
                            shape=[2, 3],
                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                            regularizer=slim.l2_regularizer(0.05)
                            )

weight2 = slim.variable('weight2',
                            shape=[2, 3],
                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                            regularizer=slim.l2_regularizer(0.05)
                            )

variables = slim.get_variables()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(weight1))
    print("--------------------")
    print(sess.run(variables))
"""


"""
# 程序19-3：slim管理用户自定义的变量
import tensorflow.contrib.slim as slim
import tensorflow as tf

weight = tf.Variable(tf.ones([2,3]))
slim.add_model_variable(weight)
model_variables = slim.get_model_variables()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(model_variables))
"""


"""
# 程序19-6：slim模型训练
import tensorflow as tf
import MLP_model as model
import tensorflow.contrib.slim as slim

save_path = './MLP_Save'

import shutil
shutil.rmtree(save_path)

g = tf.Graph()
with g.as_default():
    #在控制台打印log信息
    tf.logging.set_verbosity(tf.logging.INFO)

    #创建数据集
    xs,ys = model.produce_batch(200)
    #将数据转化为Tensor，使用这种格式能够是的Tensorflow队列自动调用
    inputs, outputs = model.convert_data_to_tensors(xs,ys)

    #计算模型值
    prediction, _ = model.mlp_model(inputs)

    #损失函数定义
    loss = slim.losses.mean_squared_error(prediction,outputs,scope="loss") #均方误差

    #使用梯度下降算法训练模型
    optimizer = slim.train.GradientDescentOptimizer(0.005)
    train_op = slim.learning.create_train_op(loss,optimizer)

    #使用Tensorflow高级执行框架“图”去执行模型训练任务。
    final_loss = slim.learning.train(
        train_op,
        logdir=save_path,
        number_of_steps=1000,
        log_every_n_steps=200,
    )

    print("Finished training. Last batch loss:", final_loss)
    print("Checkpoint saved in %s" % save_path)
"""


"""
# 程序中19-7：使用多种损失函数组合模型的训练，使用sess.run训练，而不是slim.learning.train。
import tensorflow as tf
import MLP_model as model
import tensorflow.contrib.slim as slim

save_path = './MLP_Save/'

import shutil
shutil.rmtree(save_path)

g = tf.Graph()
with g.as_default():
    #在控制台打印log信息
    tf.logging.set_verbosity(tf.logging.INFO)

    #创建数据集
    xs,ys = model.produce_batch(200)
    #将数据转化为Tensor，使用这种格式能够是的Tensorflow队列自动调用
    inputs, outputs = model.convert_data_to_tensors(xs,ys)

    #计算模型值
    prediction, end_point = model.mlp_model(inputs)

    #损失函数定义
    #均方误差
    mean_squared_error = slim.losses.mean_squared_error(prediction,outputs,scope="mean_squared_error")
    #绝对误差
    absolute_difference_loss =slim.losses.absolute_difference(prediction,outputs,scope="absolute_difference_loss")


    #定义全部的损失函数
    total_loss = mean_squared_error + absolute_difference_loss

    # 使用梯度下降算法训练模型
    optimizer = slim.train.GradientDescentOptimizer(0.005)  # 可以改成后面加mini....
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    saver = slim.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for _ in range(10000):
            _, loss_val = sess.run([train_op, total_loss])
            print(loss_val)
        saver.save(sess, save_path+"MLP_train_multiple_loss.ckpt")
        print(sess.run(end_point["fc1"]))
"""


"""
# 程序19-8：模型复用
import tensorflow as tf
import MLP_model as model
import tensorflow.contrib.slim as slim

save_path = './MLP_Save/'

with tf.Graph().as_default():
    #在控制台打印log信息
    tf.logging.set_verbosity(tf.logging.INFO)

    #创建数据集
    xs,ys = model.produce_batch(200)
    #将数据转化为Tensor，使用这种格式能够是的Tensorflow队列自动调用
    inputs, outputs = model.convert_data_to_tensors(xs,ys)

    #计算模型值
    prediction, _ = model.mlp_model(inputs,is_training=False)

    saver = tf.train.Saver()
    save_path =  tf.train.latest_checkpoint(save_path)
    with tf.Session() as sess:
        saver.restore(sess,save_path)
        inputs, prediction, outputs = sess.run([inputs,prediction,outputs])
        print(inputs, prediction, outputs)
"""


"""
# 程序19-9：MLP模型评估
import tensorflow as tf
import MLP_model as model
import tensorflow.contrib.slim as slim

save_path = './MLP_Save/'

with tf.Graph().as_default():
    #在控制台打印log信息
    tf.logging.set_verbosity(tf.logging.INFO)

    #创建数据集
    xs,ys = model.produce_batch(200)
    #将数据转化为Tensor，使用这种格式能够是的Tensorflow队列自动调用
    inputs, outputs = model.convert_data_to_tensors(xs,ys)

    #计算模型值
    prediction, _ = model.mlp_model(inputs,is_training=False)

    #制定的度量值-相对误差和绝对误差:
    names_to_value_nodes, names_to_update_nodes = slim.metrics.aggregate_metric_map({
      'Mean Squared Error': slim.metrics.streaming_mean_squared_error(prediction, outputs),
      'Mean Absolute Error': slim.metrics.streaming_mean_absolute_error(prediction, outputs)
    })

    sv = tf.train.Supervisor(logdir=save_path)
    with sv.managed_session() as sess:
        names_to_value = sess.run(names_to_value_nodes)
        names_to_update = sess.run(names_to_update_nodes)

    for key, value in names_to_value.items():
        print((key, value))

    print("\n")
    for key, value in names_to_update.items():
        print((key, value))
"""


# 程序19-10：
#encoding=utf-8
import os
import tensorflow as tf
from PIL import Image

def create_record(cwd = "C:/cat_and_dog_r",classes = {'cat',"dog"},img_heigh=224,img_width=224):
    """
    :param cwd: 主文件夹 位置 ，所有分类的数据存储在这里
    :param classes:子文件夹 名称 ，每个文件夹的名称作为一个分类，由[1,2,3......]继续分下去
    :return:最终在当前位置生成一个tfrecords文件
    """
    writer = tf.python_io.TFRecordWriter("train.tfrecords") #最终生成的文件名
    for index, name in enumerate(classes):
        class_path = cwd +"/"+ name+"/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((img_heigh, img_width))
            img_raw = img.tobytes() #将图片转化为原生bytes
            example = tf.train.Example(
               features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
               }))
            writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(filename,img_heigh=224,img_width=224):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [img_heigh, img_width, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(label, tf.int32)
    return img, label


import tensorflow as tf

def batch_read_and_decode(filename,img_heigh=224,img_width=224,batchSize=100):
    # 创建文件队列
    fileNameQue = tf.train.string_input_producer([filename], shuffle=True)
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(fileNameQue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [img_heigh, img_width, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    min_after_dequeue = batchSize * 9
    capacity = min_after_dequeue + batchSize
    # 预取图像和label并随机打乱，组成batch，此时tensor rank发生了变化，多了一个batch大小的维度
    exampleBatch,labelBatch = tf.train.shuffle_batch([img, label],batch_size=batchSize, capacity=capacity,
                                                     min_after_dequeue=min_after_dequeue)
    return exampleBatch,labelBatch


if __name__ == "__main__":
   init = tf.initialize_all_variables()
   exampleBatch, labelBatch = batch_read_and_decode("train.tfrecords")

   with tf.Session() as sess:
       sess.run(init)
       coord = tf.train.Coordinator()
       threads = tf.train.start_queue_runners(coord=coord)

       for i in range(100):
           example, label = sess.run([exampleBatch, labelBatch])
           print(example[0][112], label)
           print("---------%i---------" % i)

       coord.request_stop()
       coord.join(threads)



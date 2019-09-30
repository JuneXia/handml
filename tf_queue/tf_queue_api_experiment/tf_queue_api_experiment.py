#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


if __name__ == '__main__5':
    # 新建一个Session
    with tf.Session() as sess:
        # 我们要读三幅图片1.jpg, 2.jpg, 3.jpg
        filename = ['1.png', '2.png', '3.png']
        # string_input_producer会产生一个文件名队列
        filename_queue = tf.train.string_input_producer(filename, shuffle=False, num_epochs=5)
        # reader从文件名队列中读数据。对应的方法是reader.read
        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)
        # tf.train.string_input_producer定义了一个epoch变量，要对它进行初始化
        tf.local_variables_initializer().run()
        # 使用start_queue_runners之后，才会开始填充队列
        threads = tf.train.start_queue_runners(sess=sess)
        i = 0
        while True:
            i += 1
            # 获取图片数据并保存
            image_data = sess.run(value)
            with open('read/test_%d.jpg' % i, 'wb') as f:
                f.write(image_data)
            print(i)


# QueueRunner
if __name__ == '__main__6':
    q = tf.FIFOQueue(10, "float")
    counter = tf.Variable(0.0)  # 计数器
    # 给计数器加一
    increment_op = tf.assign_add(counter, 1.0)
    # 将计数器加入队列
    enqueue_op = q.enqueue(counter)

    # 创建QueueRunner
    # 用多个线程向队列添加数据
    # 这里实际创建了4个线程，两个增加计数，两个执行入队
    qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enqueue_op] * 2)

    # 主线程
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # 启动入队线程
    qr.create_threads(sess, start=True)
    for i in range(20):
        print(sess.run(q.dequeue()))


# Coordinator
import threading, time

# 子线程函数
def loop(coord, id):
    t = 0
    while not coord.should_stop():
        print(id)
        time.sleep(1)
        t += 1
        # 只有1号线程调用request_stop方法
        if (t >= 2 and id == 1):
            coord.request_stop()

if __name__ == '__main__7':
    # 主线程
    coord = tf.train.Coordinator()
    # 使用Python API创建10个线程
    threads = [threading.Thread(target=loop, args=(coord, i)) for i in range(10)]

    # 启动所有线程，并等待线程结束
    for t in threads: t.start()
    coord.join(threads)


# 显式的创建QueueRunner，然后调用它的create_threads方法启动线程
if __name__ == '__main__8':
    # 1000个4维输入向量，每个数取值为1-10之间的随机数
    data = 10 * np.random.randn(1000, 4) + 1
    # 1000个随机的目标值，值为0或1
    target = np.random.randint(0, 2, size=1000)

    # 创建Queue，队列中每一项包含一个输入数据和相应的目标值
    queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.int32], shapes=[[4], []])

    # 批量入列数据（这是一个Operation）
    enqueue_op = queue.enqueue_many([data, target])
    # 出列数据（这是一个Tensor定义）
    data_sample, label_sample = queue.dequeue()

    # 创建包含4个线程的QueueRunner
    qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)

    with tf.Session() as sess:
        print(queue.size().eval())
        # 创建Coordinator
        coord = tf.train.Coordinator()
        # 启动QueueRunner管理的线程
        enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
        print(queue.size().eval())

        # 主线程，消费100个数据
        for step in range(1000):
            if coord.should_stop():
                break
            data_batch, label_batch = sess.run([data_sample, label_sample])
            print(step, data_batch, label_batch)
        # 主线程计算完成，停止所有采集数据的进程
        coord.request_stop()
        coord.join(enqueue_threads)


# 使用全局的start_queue_runners方法启动线程
if __name__ == '__main__9':
    # 同时打开多个文件，显示创建Queue，同时隐含了QueueRunner的创建
    filename_queue = tf.train.string_input_producer(["data1.csv", "data2.csv"])
    reader = tf.TextLineReader(skip_header_lines=1)
    # Tensorflow的Reader对象可以直接接受一个Queue作为输入
    key, value = reader.read(filename_queue)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        # 启动计算图中所有的队列线程
        threads = tf.train.start_queue_runners(coord=coord)
        print(filename_queue.size().eval())

        # 主线程，消费100个数据
        for step in range(100):
            features, labels = sess.run([key, value])
            print(step, features, labels)
        # 主线程计算完成，停止所有采集数据的进程
        coord.request_stop()
        coord.join(threads)


# 使用全局的tf.train.start_queue_runners启动队列。
def generate_data():
    num = 25
    label = np.asarray(range(0, num))
    images = np.random.random([num, 5])
    print('label size :{}, image size {}'.format(label.shape, images.shape))
    return images,label


def get_batch_data():
    images, label = generate_data()
    input_queue = tf.train.slice_input_producer([images, label], shuffle=False,num_epochs=2)
    image_batch, label_batch = tf.train.batch(input_queue, batch_size=5, num_threads=1, capacity=64, allow_smaller_final_batch=False)
    return image_batch,label_batch


if __name__ == '__main__10':
    images, label = get_batch_data()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())  # 这一行必须加，因为slice_input_producer的原因
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    try:
        count = 0
        while not coord.should_stop():
            i, l = sess.run([images, label])
            print(count, i.shape, l.shape)
            count += 1
    except tf.errors.OutOfRangeError:
        print('Done training')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


# 单个Reader, 单个样本
if __name__ == '__main__11':
    # 生成一个先入先出队列和一个QueueRunner
    filenames = ['data1.csv', 'data2.csv', 'data3.csv']
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
    # 定义Reader
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    # 定义Decoder
    # example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])
    example, label = tf.decode_csv(value, record_defaults=[['string'], ['string']])
    # 运行Graph
    with tf.Session() as sess:
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord)  # 启动QueueRunner, 此时文件名队列已经进队。
        count = 0
        while True:
            # print(count, example.eval(), label.eval())  # 取样本的时候，一个Reader先从文件名队列中取出文件名，读出数据，Decoder解析后进入样本队列。
            print(count, sess.run([example, label]))
            count += 1
        coord.request_stop()
        coord.join(threads)


# 单个Reader, 多个样本
if __name__ == '__main__12':
    filenames = ['data1.csv', 'data2.csv', 'data3.csv']
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    example, label = tf.decode_csv(value, record_defaults=[['string'], ['string']])
    # 使用tf.train.batch()会多加了一个样本队列和一个QueueRunner。Decoder解后数据会进入这个队列，再批量出队。
    # 虽然这里只有一个Reader，但可以设置多线程，相应增加线程数会提高读取速度，但并不是线程越多越好。
    example_batch, label_batch = tf.train.batch([example, label], batch_size=2)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        count = 0
        while True:
            print(count, sess.run([example_batch, label_batch]))
            count += 1
        coord.request_stop()
        coord.join(threads)


# 多个Reader, 多个样本
if __name__ == '__main__13':
    filenames = ['data1.csv', 'data2.csv', 'data3.csv']
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [['string'], ['string']]
    example_list = [tf.decode_csv(value, record_defaults=record_defaults) for _ in range(2)]  # Reader设置为2
    # 使用tf.train.batch_join()，可以使用多个reader，并行读取数据。每个Reader使用一个线程。
    example_batch, label_batch = tf.train.batch_join(example_list, batch_size=2)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        count = 0
        while True:
            print(count, sess.run([example_batch, label_batch]))
            count += 1
        coord.request_stop()
        coord.join(threads)


# 迭代控制
if __name__ == '__main__':
    filenames = ['data1.csv', 'data2.csv', 'data3.csv']
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False, num_epochs=3)  # num_epoch: 设置迭代数
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [['string'], ['string']]
    example_list = [tf.decode_csv(value, record_defaults=record_defaults)
                    for _ in range(2)]
    example_batch, label_batch = tf.train.batch_join(example_list, batch_size=2)
    init_local_op = tf.initialize_local_variables()
    with tf.Session() as sess:
        sess.run(init_local_op)  # 初始化本地变量
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            count = 0
            while not coord.should_stop():
                print(count, sess.run([example_batch, label_batch]))
                count += 1
        except tf.errors.OutOfRangeError:
            print('Epochs Complete!')
        finally:
            coord.request_stop()
        coord.join(threads)
        coord.request_stop()
        coord.join(threads)







queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32], shapes=[[4]])

enqueue_op = queue.enqueue_many([input_data])
data_sample = queue.dequeue()

# 创建包含4个线程的QueueRunner
qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)

with tf.Session() as sess:
    # 创建线程协调器Coordinator
    coord = tf.train.Coordinator()
    # 启动线程
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

    # 主线程，消费数据
    for step in range(1000):
        if coord.should_stop():
            break
        data_batch = sess.run([data_sample])
        print(step, data_batch)
    # 主线程计算完成，停止所有采集数据的进程
    coord.request_stop()
    coord.join(enqueue_threads)
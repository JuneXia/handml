"""
# 程序20-6：载入预训练模型
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import inception_resnet_v2 as model

checkpoints_dir = './pretrain_model'
with tf.Graph().as_default():
    img = tf.random_normal([1, 299, 299, 3])

    with slim.arg_scope(model.inception_resnet_v2_arg_scope()):
        pre,_ = model.inception_resnet_v2(img, num_classes=1001, is_training=False)

    model_path = os.path.join(checkpoints_dir, 'inception_resnet_v2_2016_08_30.ckpt')
    variables = slim.get_model_variables('InceptionResnetV2')
    init_fn = slim.assign_from_checkpoint_fn(model_path,variables)

    with tf.Session() as sess:
       init_fn(sess)
       print( (sess.run(pre)))
       print("done")
"""


"""
# 程序20-7：使用预训练模型，不改变class_num的情况下尝试训练自己的数据集
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import inception_resnet_v2 as model

checkpoints_dir = './pretrain_model'
flowers_data_dir = './flowers/'

# 载入数据的函数
def load_batch(dataset, batch_size=4, height=299, width=299, is_training=True):
    import tensorflow.contrib.slim.python.slim.data.dataset_data_provider as providerr
    data_provider = providerr.DatasetDataProvider(dataset, common_queue_capacity=8, common_queue_min=1)
    image_raw, label = data_provider.get(['image', 'label'])
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.image.convert_image_dtype(image_raw, tf.float32)

    images_raw, labels = tf.train.batch(
        [image_raw, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=2 * batch_size)
    return images_raw, labels


# 训练模型
fintuning = tf.Graph()
with fintuning.as_default():
    tf.logging.set_verbosity(tf.logging.INFO)

    # 获取数据集
    from datasets import flowers

    dataset = flowers.get_split('train', flowers_data_dir)
    images, labels = load_batch(dataset)

    # 载入模型，此时模型未载入参数
    with slim.arg_scope(model.inception_resnet_v2_arg_scope()):
        pre, _ = model.inception_resnet_v2(images, num_classes=1001)
    probabilities = tf.nn.softmax(pre)

    # 对标签进行格式化处理
    one_hot_labels = slim.one_hot_encoding(labels, 1001)

    # 创建损失函数
    slim.losses.softmax_cross_entropy(probabilities, one_hot_labels)
    total_loss = slim.losses.get_total_loss()

    # 创建训练节点
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    # 准备载入模型权重的函数
    model_path = os.path.join(checkpoints_dir, 'inception_resnet_v2_2016_08_30.ckpt')
    variables = slim.get_model_variables('InceptionResnetV2')
    init_fn = slim.assign_from_checkpoint_fn(model_path, variables)

    # 正式载入模型权重并开始训练
    with tf.Session() as sess:
        init_fn(sess)
        print("done")
    final_loss = slim.learning.train(
        train_op,
        logdir=None,
        number_of_steps=10
    )
"""


"""
# 程序20-8：通过增加一个全连接层来改变输出类别，从而训练自己的模型
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import inception_resnet_v2 as model

checkpoints_dir = './pretrain_model'
flowers_data_dir = './flowers/'
save_path = './save_model'

def load_batch(dataset, batch_size=8, height=299, width=299, is_training=True):

    import tensorflow.contrib.slim.python.slim.data.dataset_data_provider as providerr
    data_provider = providerr.DatasetDataProvider(
        dataset, common_queue_capacity=8, common_queue_min=1)
    image_raw, label = data_provider.get(['image', 'label'])
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.image.convert_image_dtype(image_raw,tf.float32)

    images_raw, labels = tf.train.batch(
        [image_raw, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=2 * batch_size)
    return images_raw, labels

fintuning_newFC = tf.Graph()
with fintuning_newFC.as_default():
    tf.logging.set_verbosity(tf.logging.INFO)

    from datasets import flowers
    dataset = flowers.get_split('train', flowers_data_dir)
    images, labels = load_batch(dataset)

    with slim.arg_scope(model.inception_resnet_v2_arg_scope()):
        pre,_ = model.inception_resnet_v2(images,num_classes=1001)
        pre = slim.fully_connected(pre,5)
    probabilities = tf.nn.softmax(pre)

    one_hot_labels = slim.one_hot_encoding(labels, 5)

    slim.losses.softmax_cross_entropy(probabilities, one_hot_labels)
    total_loss = slim.losses.get_total_loss()

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    model_path = os.path.join(checkpoints_dir, 'inception_resnet_v2_2016_08_30.ckpt')
    variables = slim.get_model_variables('InceptionResnetV2')
    init_fn = slim.assign_from_checkpoint_fn(model_path, variables)

    with tf.Session() as sess:
        init_fn(sess)
        print("done")
        final_loss = slim.learning.train(
                train_op,
                logdir=save_path,
                number_of_steps=100
        )
"""


# 程序20-9：从头开始训练自己的模型
# 代码略


# 程序20-10：
import os
import tensorflow as tf
import inception_resnet_v2 as model
import tensorflow.contrib.slim as slim


checkpoints_dir = './pretrain_model'
flowers_data_dir = './flowers/'
save_path = './save_model'

image_size = 299

def get_init_fn(sess):
    #不进行载入的layer
    checkpoint_exclude_scopes = ["InceptionResnetV2/AuxLogits","InceptionResnetV2/Logits"]
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    model_path = os.path.join(checkpoints_dir, 'inception_resnet_v2_2016_08_30.ckpt')
    init_fn = slim.assign_from_checkpoint_fn(model_path, variables_to_restore)

    return init_fn(sess)


with tf.Graph().as_default():
    img = tf.random_normal([1, 299, 299, 3])

    with slim.arg_scope(model.inception_resnet_v2_arg_scope()):
        pre,_ = model.inception_resnet_v2(img, is_training=False)

    with tf.Session() as sess:

        init_fn = get_init_fn(sess)
        res = (sess.run(pre))
        print(res.shape)
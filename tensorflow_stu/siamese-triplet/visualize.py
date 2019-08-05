# coding:utf-8
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import matplotlib.pyplot as plt
from lenet_training import lenet
import os


# 自定义sprite文件和meta文件，创建每个点使用的小图
def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as
     argument. Images should be count x width x height  """
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = this_img

    return spriteimage


def sprite_and_meta_writer(batch_xs, batch_ys):
    to_visualise = np.reshape(batch_xs, (-1, 32, 32))  # 转换图片形状为3D
    to_visualise = 1 - to_visualise  # 转换背景颜色为白色

    sprite_image = create_sprite_image(to_visualise)

    plt.imsave(os.path.join(LOG_DIR, 'mnistdigits.png'), sprite_image, cmap='gray')

    # 创建 metadata.tsv 的标签 文件
    with open(os.path.join(LOG_DIR, 'metadata.tsv'), 'w') as f:
        # f.write("Index\tLabel\n")
        # 但是注意 Single column metadata should not have a header row. 单列的metadata 不能有头文字描述
        for index, label in enumerate(batch_ys):
            # 按照这个被注释的指定方式，每个点上面悬浮的是它的序号索引值，如果只写label，那么每个点上悬浮的是它的label
            # f.write("%d\t%d\n" % (index, label))   # 'metadata.tsv'  就是一个 序号index加label的两列数据文件
            f.write("%d\n" % (label))  # 'metadata.tsv'  就是一个 序号index加label的两列数据文件


if __name__ == '__main__':
    """
    这个可视化的操作实际上最简单的只需要两个步骤：
    1 指定需要可视化的2D变量（不指定也可以，会显示默认可以显示的2D变量）
    embedding_var = tf.Variable(xxx, name='xxx')  # 需要可视化的变量，只能是2D的

    2 保存模型的结果，然后在tensorboard里面就可以看到每个2D变量经过降维之后的展示
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(LOG_DIR, "embeding_model.ckpt"), 1)

    3 （可选）给embedding_var关联metadata文件，这样我们就可以知道每个点的标签了

    """

    mnist = input_data.read_data_sets(r"C:\Users\lon\Documents\slowlon\TFdemo\mnist\Mnist_data", reshape=False,
                                      one_hot=True)

    # 配置
    LOG_DIR = 'checkpoints'
    batch_size = 512

    # 定义网络结构
    net = lenet(0.001, False)
    # 导入模型变量值
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'checkpoints/lenet.ckpt')

        # 获取要展示的图片数据
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        # pading 图片
        x_batch = np.pad(batch_xs, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
        # 创建需要的 metadata.tsv 标注文件和每个点需要的图片文件
        sprite_and_meta_writer(x_batch, np.argmax(batch_ys, axis=1))

        # 获取第二个全连接层的特征作为展示，也可以换为其他的
        fc2_samples = sess.run([net.fc2], feed_dict={
            net.x: x_batch,
            net.label: batch_ys, })[0]

        # *****可视化的设置*****
        embedding_var = tf.Variable(fc2_samples, name='fc2_samples')  # 需要可视化的变量，只能是2D的
        sess.run(embedding_var.initializer)  # 使用之前还需要再初始化一次

        summary_writer = tf.summary.FileWriter(LOG_DIR)

        # 建立 embedding projector：指定想要可视化的 variable 的 metadata 文件的位置
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name

        # Specify where you find the metadata
        embedding.metadata_path = 'metadata.tsv'  # 这里必须写成相对路径 ，
        # 因为它是在相对于 tensorboard logdir 的相对路径中寻找的metadata文件，所以不能传入绝对路径，
        # 下面的mnistdigits.png也一样，否则会出现 chrome，  parsing metadata 界面，
        # 但是一直卡住没有进展，这就是因为它找不到这个metadata.tsv文件导致的

        # Specify where you find the sprite (we will create this later)
        # 这个图片是每个数据点使用的小图片，但是这里是生成了一张大图包含了这么多张小图，
        # 而这个小图是怎么分配到每个点的还是不是很清楚,猜测是通过single_image_dim来逐个读取的,
        # 类似于 label 的 metadata逐个读取方式
        embedding.sprite.image_path = 'mnistdigits.png'
        embedding.sprite.single_image_dim.extend([32, 32])

        # Say that you want to visualise the embeddings 保存配置文件
        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(summary_writer, config)

        saver = tf.train.Saver()  # 保存模型文件，注意和保存配置文件在一个文件夹
        saver.save(sess, os.path.join(LOG_DIR, "embeding_model.ckpt"), 1)

        # 然后在保存路径打开 tensorboard 就可以看到效果了。

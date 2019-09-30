# 19-4：MLP的slim实现
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt

def mlp_model(inputs,is_training=True, scope="mlp_model"):
    """
    创建一个mlp模型
    :param input: 一个大小为[batch_size,dimensions]的Tensor张量作为输入数据；
    :param is_training: 是否模型处于训练状态。（当进行使用时模型处于非训练状态，计算时可节省大量时间）
    :param scope:命名空间的名称
    :return:prediction,end_point。其中prediction是模型计算的最终值，而end_point用以收集每层计算值的字典。
    """
    with tf.variable_scope(scope,"mlp_model",[inputs]):
        #使用end_point记录每一层的输出，这样做的好处是对每一层的输出都有个记录，方便在后期进行Fintuning。
        end_point = {}
        #创建一个参数空间用以记录使用的各种层和激活函数，以及各种参数的正则化修正。
        with slim.arg_scope(
                [slim.fully_connected],
                activation_fn = tf.nn.relu,
                weights_regularizer = slim.l2_regularizer(0.01)
        ):

        #第一个全连接层，输出为32个节点，
        # 这里需要注意的是全连接层中所需要的参数的定义，激活函数的使用在前面的arg_scope已经定义过。
            net = slim.fully_connected(inputs,32,scope="fc1")
            end_point["fc1"] = net
        #使用dropout进行全连接层修正，每次保存的数目为0.5。
            net = slim.dropout(net,0.5,is_training=is_training)

        #第二个全连接层，输出为16个节点。
            net = slim.fully_connected(net,16,scope="fc2")
            end_point["fc2"] = net
        #使用dropout进行全连接层修正，每次保存的数目为0.5。
            net = slim.dropout(net,0.5,is_training=is_training)

        #使用一个全连接层作为最终层的计算，输出1个值，不使用激活函数。
            prediction = slim.fully_connected(net,1,activation_fn=None,scope = "prediction")
            end_point["out"] = prediction
        return prediction,end_point


# 程序19-5：创建数据集
def produce_batch(batch_size, noise=0.3):
    xs = np.random.random(size=[batch_size, 1]) * 10
    ys = np.cos(xs) + 5 + np.random.normal(size=[batch_size, 1], scale=noise)
    return [xs.astype(np.float32), ys.astype(np.float32)]


def convert_data_to_tensors(x, y):
    inputs = tf.constant(x)
    inputs.set_shape([None, 1])
    outputs = tf.constant(y)
    outputs.set_shape([None, 1])
    return inputs, outputs


if __name__ == "__main__" :
    x_train, y_train = produce_batch(200)
    x_test, y_test = produce_batch(200)
    plt.scatter(x_train,y_train,marker="8")
    plt.scatter(x_test, y_test,marker="*")
    plt.show()
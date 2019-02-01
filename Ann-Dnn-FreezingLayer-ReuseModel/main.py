import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.examples.tutorials.mnist.input_data as input_data


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


datapath = "/home/xiajun/res/MNIST_data"
mnist = input_data.read_data_sets(datapath, validation_size=0, one_hot=True)

with tf.Graph().as_default():
    with tf.Session() as sess:
        # load_model('./my_model_final_no_dnnscope')  # OK

        saver = tf.train.import_meta_graph(
            '/home/xiajun/devstu/handml/my_model_final/my_model_final.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./my_model_final'))

        ''
        # 如果在下面的 get_tensor_by_name 中不知道张量名，则可以通过这里的打印看出来。
        for op in tf.get_default_graph().as_graph_def().node:
            print(op.name)
        ''

        X_placeholder = tf.get_default_graph().get_tensor_by_name('X:0')
        yy = tf.get_default_graph().get_tensor_by_name('yy:0')
        # logits = tf.get_default_graph().get_tensor_by_name('outputs:0')  # error
        logits = tf.get_default_graph().get_tensor_by_name('dnn/outputs/add:0')
        accuracy = tf.get_default_graph().get_tensor_by_name('eval/Mean:0')

        X_batch = mnist.test.images
        y_batch = mnist.test.labels
        yy_batch = np.argmax(y_batch, 1)
        acc_rate = accuracy.eval(feed_dict={X_placeholder: X_batch, yy: yy_batch})

        # 上面是模型训练时定义的accuracy评测标准，实际使用时也可以用下面的方法进行预测
        predict_array = sess.run(logits, feed_dict={X_placeholder:X_batch})
        # predict_array = logits.eval(feed_dict={X_placeholder: X_new_scaled})  # 效果同上
        y_pred = np.argmax(predict_array, axis=1)
        acc_rate = np.mean(np.equal(y_pred, yy_batch))
        print(y_pred)
        print(acc_rate)

print('end')

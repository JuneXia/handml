import numpy as np
import matplotlib.pyplot as plt

csv_list = [
    # 'AlexNet-input?x227x227x1-convBN?x55x55x96-maxpool?x27x27x96-convBN?x27x27x256-maxpool?x13x13x256-convBN?x13x13x384-convBN?x13x13x384-convBN?x13x13x256-maxpool?x6x6x256-fcBN?x4096-fcBN?x4096-output?x10-learnrate_1e-4-customInit_DefaultParam-epoch50.csv',
    'AlexNet2-input?x227x227x1-conv?x55x55x96lrn-maxpool?x27x27x96-conv?x27x27x256lrn-maxpool?x13x13x256-conv?x13x13x384-conv?x13x13x384-conv?x13x13x256-maxpool?x6x6x256-fc?x4096-fc?x4096-output?x10-learnrate_1e-4-customInit-epoch50.csv',  # AlexNet2.py
    'AlexNet3-input?x227x227x1-conv?x55x55x96lrn-maxpool?x27x27x96-conv?x27x27x256lrn-maxpool?x13x13x256-conv?x13x13x384-conv?x13x13x384-conv?x13x13x256-maxpool?x6x6x256-fc?x4096-fc?x4096-output?x10-learnrate_feed-customInit-epoch50.csv',  # AlexNet3.py
    'AlexNet4-input?x227x227x1-conv?x55x55x96lrn-maxpool?x27x27x96-conv?x27x27x256lrn-maxpool?x13x13x256-conv?x13x13x384-conv?x13x13x384-conv?x13x13x256-maxpool?x6x6x256-fc?x4096-fc?x4096-output?x10-learnrate_1e-4-tf_global_init-epoch50.csv',  # AlexNet4.py
    'AlexNet5-input?x227x227x1-conv?x55x55x96lrn-maxpool?x27x27x96-conv?x27x27x256lrn-maxpool?x13x13x256-conv?x13x13x384-group2conv?x13x13x384-group2conv?x13x13x256-maxpool?x6x6x256-fc?x4096-fc?x4096-output?x10-learnrate_1e-4-customInit-epoch50.csv',  # AlexNet5.py
    'AlexNet6-input?x227x227x1-convBN?x55x55x96-maxpool?x27x27x96-convBN?x27x27x256-maxpool?x13x13x256-convBN?x13x13x384-convBN?x13x13x384-convBN?x13x13x256-maxpool?x6x6x256-fcBN?x4096-fcBN?x4096-output?x10-learnrate_1e-4-customInit-epoch50.csv',  # AlexNet6.py
]

csv_label = [
    'AlexNet2-customInit',
    'AlexNet3-feed_learningrate',
    'AlexNet4-tf_global_init',
    'AlexNet5-group_conv',
    'AlexNet6-BN'
]

val_buf = []
for file_name in csv_list:
    print(file_name)
    file_path = '../AlexNet/' + file_name
    fid = open(file_path, 'r')

    tmp_buf = []
    for val in fid.readlines():
        # print(val)
        tmp_buf.append(val.strip().split(','))
    val_buf.append(tmp_buf)
    fid.close()


val_array = np.array(val_buf)
val_array = val_array.astype(np.float)

col = 1  # 0: loss, 1: accuracy rate
step = 1
plt.plot(np.arange(val_array[0][0::step].shape[0]), val_array[0][0::step, col], color='green', label=csv_label[0])
plt.plot(np.arange(val_array[1][0::step].shape[0]), val_array[1][0::step, col], color='red', label=csv_label[1])
plt.plot(np.arange(val_array[2][0::step].shape[0]), val_array[2][0::step, col], color='blue', label=csv_label[2])
plt.plot(np.arange(val_array[3][0::step].shape[0]), val_array[3][0::step, col], color='skyblue', label=csv_label[3])
plt.plot(np.arange(val_array[4][0::step].shape[0]), val_array[4][0::step, col], color='yellow', label=csv_label[4])
# plt.plot(np.arange(val_array[5][0::step].shape[0]), val_array[5][0::step, 0], color='purple', label=csv_label[5])
plt.legend()  # 显示label
plt.xlabel('iteration times')
plt.ylabel('accuracy rate')
plt.show()

print('end')


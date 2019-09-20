import numpy as np
import matplotlib.pyplot as plt

csv_list = [
    'LeNet-input?x28x28x1-conv?x28x28x6_maxpool?x14x14x6-conv?x14x14x16_maxpool?x10x10x16-conv?x5x5x120-fc?x1024-output?x10-epoch2000.csv',
    'LeNet-input?x28x28x1-conv?x28x28x6_maxpool?x14x14x6-conv?x14x14x16_maxpool?x10x10x16-conv?x5x5x120-fc?x84-output?x10-epoch2000.csv',
    'LeNet-input?x28x28x1-conv?x28x28x6_maxpool?x14x14x6-conv?x14x14x16_maxpool?x7x7x16-conv?x7x7x120-fc?x1024-output?x10-epoch2000.csv',
    'LeNet-input?x28x28x1-conv?x28x28x6_maxpool?x14x14x6-conv?x14x14x16_maxpool?x7x7x16-conv?x7x7x120-fc?x84-output?x10-epoch2000.csv',
    'LeNet-input?x28x28x1-conv?x28x28x6_maxpool?x14x14x6lrn-conv?x14x14x16lrn_maxpool?x10x10x16-conv?x5x5x120-fc?x84-output?x10-epoch2000.csv',
    'LeNet-input?x28x28x1-conv?x28x28x6_maxpool?x14x14x6lrn-conv?x14x14x16_maxpool?x10x10x16lrn-conv?x5x5x120-fc?x84-output?x10-epoch2000.csv'
]

csv_label = [
    'conv5x5x120fc1024',
    'conv5x5x120fc84',
    'conv7x7x120fc1024',
    'conv7x7x120fc84',
    'conv5x5x120fc84convlrn',
    'conv5x5x120fc84poollrn'
]

val_buf = []
for file_name in csv_list:
    print(file_name)
    file_path = '../' + file_name
    fid = open(file_path, 'r')

    tmp_buf = []
    for val in fid.readlines():
        # print(val)
        tmp_buf.append(val.strip().split(','))
    val_buf.append(tmp_buf)
    fid.close()


val_array = np.array(val_buf)
val_array = val_array.astype(np.float)

step = 1
plt.plot(np.arange(val_array[0][0::step].shape[0]), val_array[0][0::step, 0], color='green', label=csv_label[0])
plt.plot(np.arange(val_array[1][0::step].shape[0]), val_array[1][0::step, 0], color='red', label=csv_label[1])
plt.plot(np.arange(val_array[2][0::step].shape[0]), val_array[2][0::step, 0], color='blue', label=csv_label[2])
plt.plot(np.arange(val_array[3][0::step].shape[0]), val_array[3][0::step, 0], color='skyblue', label=csv_label[3])
plt.plot(np.arange(val_array[4][0::step].shape[0]), val_array[4][0::step, 0], color='yellow', label=csv_label[4])
plt.plot(np.arange(val_array[5][0::step].shape[0]), val_array[5][0::step, 0], color='purple', label=csv_label[5])
plt.legend()  # 显示label
plt.xlabel('iteration times')
plt.ylabel('loss')
plt.show()

print('end')


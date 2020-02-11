import os
import shutil
import numpy as np

lst = [5, 2, 6, 9, 1]
lst.pop(1)


if __name__ == '__main__':
    data_path = '/home/xiaj/dev/TensorFlow-2.x-Tutorials-master/深度学习与TensorFlow入门实战-源码和PPT'
    save_path = '/home/xiaj/dev/TensorFlow-2.x-Tutorials-master/save_pdf'
    clses_path = os.listdir(data_path)
    for cls in clses_path:
        cls_path = os.path.join(data_path, cls)
        if os.path.isdir(cls_path):
            files_path = os.listdir(cls_path)
            for file in files_path:
                file_path = os.path.join(cls_path, file)
                if file_path.endswith('.pdf'):
                    save_file = cls + '-' + file
                    file_save_path = os.path.join(save_path, save_file)
                    if os.path.exists(file_save_path):
                        raise Exception('不应该出现有相同名字的pdf文件的，需要排查，需要手动修改pdf名字！')
                    shutil.copy(file_path, file_save_path)

    print('debug')

# -*- coding: UTF-8 -*-
import os
import math
import sys
import skimage
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc
import shutil
from sklearn.manifold import TSNE


DEBUG = True


def view_bar(message, num, total):
    """
    :param message:
    :param num:
    :param total:
    :return:

    Example:
    view_bar('loading: ', i + 1, len(list))
    """
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()


def show_rect(img_path, regions):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    img = skimage.io.imread(img_path)
    ax.imshow(img)
    for x, y, w, h in regions:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.show()


def strcat(strlist, cat_mark=','):
    '''
    :param strlist: a list for store string.
    :param cat_mark:
    :return:
    '''
    line = ''
    for ln in strlist:
        line += str(ln) + cat_mark
    line = line[0:line.rfind(cat_mark)]
    return line


def imcrop(img, bbox, scale_ratio=2.0):
    '''
    :param img: ndarray or image_path
    :param bbox: bounding box
    :param scale_ratio: the scale_ratio used to control margin size around bbox.
    :return:
    '''
    if type(img) == str:
        img = cv2.imread(img)

    xmin, ymin, xmax, ymax = bbox
    hmax, wmax, _ = img.shape
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    w = (xmax - xmin) * scale_ratio
    h = (ymax - ymin) * scale_ratio

    xmin = x - w / 2
    xmax = x + w / 2
    ymin = y - h / 2
    ymax = y + h / 2

    xmin = max(0, int(xmin))
    ymin = max(0, int(ymin))
    xmax = min(wmax, int(xmax))
    ymax = min(hmax, int(ymax))

    face = img[ymin:ymax, xmin:xmax, :]
    return face


def named_standard(data_path, mark='', replace=''):
    class_list = os.listdir(data_path)
    for i, cls in enumerate(class_list):
        if cls.count(mark):
            new_cls = cls.replace(mark, replace)
            class_path = os.path.join(data_path, cls)
            new_class_path = os.path.join(data_path, new_cls)
            os.rename(class_path, new_class_path)
            print('[tools.named_standard]:: change {} to {}'.format(class_path, new_class_path))

        view_bar('[tools.load_image]:: loading: ', i + 1, len(class_list))


def load_image(data_path, subdir='', min_num_image_per_class=1, del_under_min_num_class=False, min_area4del=0):
    '''
    :param data_path:
    :param subdir: 如果每个类别又子目录，则应该指定。
    :param min_num_image_per_class: 每个类至少min_num_image_per_class张图片，少于这个数量的类不会被加载。
    :param del_under_min_num_class: if True, 如果每个类别少于min_num_image_per_class张图片，则永久删除这个类。
    :param min_area4del: 如果图片尺寸小于该值则会被删除。设置为0即不适用该操作。
    :return:
    '''
    del_count = 0

    images_info = []
    class_list = os.listdir(data_path)
    if DEBUG:
        class_list = class_list[0:100]
    class_list.sort()
    for i, cls in enumerate(class_list):
        class_images_info = []
        class_path = os.path.join(data_path, cls)
        image_list = os.listdir(class_path)
        image_list.sort()
        for image in image_list:
            image_path = os.path.join(class_path, image)
            if os.path.isdir(image_path):
                if image == subdir:
                    sub_image_list = os.listdir(image_path)
                    sub_image_list.sort()
                    for sub_image in sub_image_list:
                        sub_image_path = os.path.join(image_path, sub_image)
                        if os.path.isfile(sub_image_path):
                            if min_area4del > 0:
                                img = cv2.imread(sub_image_path)
                                img_area = img.shape[0] * img.shape[1]
                                if img_area < min_area4del:
                                    os.remove(sub_image_path)
                                    del_count += 1

                            sub_image = os.path.join(image, sub_image)
                            class_images_info.append([cls, sub_image])
                        else:
                            # raise Exception("Error, Shouldn't exist! Please check your param or trying to reason for {}!".format(sub_image_path))
                            print(image_path)
                            if os.path.exists(class_path):
                                shutil.rmtree(class_path)
                else:
                    # raise Exception("Error, Shouldn't exist! Please check your param or trying to reason for {}!".format(image_path))
                    print(image_path)
                    if os.path.exists(class_path):
                        shutil.rmtree(class_path)
            else:
                if min_area4del > 0:
                    img = cv2.imread(image_path)
                    img_area = img.shape[0] * img.shape[1]
                    if img_area < min_area4del:
                        os.remove(image_path)
                        del_count += 1

                class_images_info.append([cls, image])

        if del_under_min_num_class and len(class_images_info) < min_num_image_per_class:
            if os.path.exists(class_path):
                del_count += 1
                shutil.rmtree(class_path)
            print('[tools.load_image]:: delete: {} image, {}'.format(len(class_images_info), class_path))
            continue
        elif del_under_min_num_class and len(cls) > 30:
            # del_count += 1
            # shutil.rmtree(class_path)
            print('[tools.load_image]:: delete: {} image, {}'.format(len(class_images_info), class_path))
        else:
            images_info.extend(class_images_info)

        view_bar('[tools.load_image]:: loading: ', i + 1, len(class_list))
    print('')
    print('[tools.load_image]:: delete {} images!'.format(del_count))
    return images_info


def compute_auc(fpr, tpr):
    roc_auc = auc(fpr, tpr)
    return roc_auc


def _get_plt_color(index=None):
    """
    num_class = 50
    cmap = plt.get_cmap('gnuplot')
    # 方法1
    colors = [cmap(i) for i in np.linspace(0, 1, num_class)]
    # 方法2
    colors = plt.cm.rainbow(np.linspace(0.0, 1.0, num_class))

    # 方法3
    cnames = plt.cm.colors.cnames
    colors = []
    for color in cnames:
        colors.append(cnames[color])

    # 方法4
    colors = list(matplotlib.colors.cnames.keys())
    colors.sort()
    """

    colors = [
        # 'aliceblue',
        'antiquewhite',
        'aqua',
        'aquamarine',
        ##'azure',
        'beige',
        'bisque',
        'black',
        'blanchedalmond',
        'blue',
        'blueviolet',
        'brown',
        'burlywood',
        'cadetblue',
        'chartreuse',
        'chocolate',
        'coral',
        'cornflowerblue',
        # 'cornsilk',
        'crimson',
        'cyan',
        'darkblue',
        'darkcyan',
        'darkgoldenrod',
        # 'darkgray',  # 和 darkgrey 有点像
        'darkgreen',
        'darkgrey',
        'darkkhaki',
        'darkmagenta',
        'darkolivegreen',
        'darkorange',
        'darkorchid',
        'darkred',
        'darksalmon',
        'darkseagreen',
        'darkslateblue',
        'darkslategray',
        'darkslategrey',
        'darkturquoise',
        'darkviolet',
        'deeppink',
        'deepskyblue',
        'dimgray',
        'dimgrey',
        'dodgerblue',
        'firebrick',
        # 'floralwhite',
        'forestgreen',
        'fuchsia',
        'gainsboro',
        # 'ghostwhite',
        'gold',
        'goldenrod',
        'gray',
        'green',
        'greenyellow',
        'grey',
        # 'honeydew',
        'hotpink',
        'indianred',
        'indigo',
        # 'ivory',
        'khaki',
        'lavender',
        'lavenderblush',
        'lawngreen',
        'lemonchiffon',
        'lightblue',
        'lightcoral',
        'lightcyan',
        'lightgoldenrodyellow',
        'lightgray',
        'lightgreen',
        'lightgrey',
        'lightpink',
        'lightsalmon',
        'lightseagreen',
        'lightskyblue',
        'lightslategray',
        'lightslategrey',
        'lightsteelblue',
        # 'lightyellow',
        'lime',
        'limegreen',
        'linen',
        'magenta',
        'maroon',
        'mediumaquamarine',
        'mediumblue',
        'mediumorchid',
        'mediumpurple',
        'mediumseagreen',
        'mediumslateblue',
        'mediumspringgreen',
        'mediumturquoise',
        'mediumvioletred',
        # 'midnightblue',
        'mintcream',
        'mistyrose',
        'moccasin',
        'navajowhite',
        'navy',
        # 'oldlace',
        'olive',
        'olivedrab',
        'orange',
        'orangered',
        'orchid',
        'palegoldenrod',
        'palegreen',
        'paleturquoise',
        'palevioletred',
        'papayawhip',
        'peachpuff',
        'peru',
        'pink',
        'plum',
        'powderblue',
        'purple',
        'rebeccapurple',
        'red',
        'rosybrown',
        'royalblue',
        'saddlebrown',
        'salmon',
        'sandybrown',
        'seagreen',
        # 'seashell',
        'sienna',
        'silver',
        'skyblue',
        'slateblue',
        # 'slategray',  # 和slategrey有点像
        'slategrey',
        'snow',
        'springgreen',
        'steelblue',
        'tan',
        'teal',
        'thistle',
        'tomato',
        'turquoise',
        'violet',
        'wheat',
        # 'white',
        # 'whitesmoke',
        'yellow',
        'yellowgreen']

    if (index is None) or (index >= len(colors)):
        index = np.random.randint(0, len(colors))

    return colors[index]


def _get_plt_linestyle(index=None):
    # linestyle = matplotlib.style.available
    linestyle = [#'.',
                 #',',
                 #'o',
                 #'v',
                 #'^',
                 #'<',
                 #'>',
                 #'1',
                 #'2',
                 #'3',
                 #'4',
                 #'s',
                 #'p',
                 #'*',
                 #'h',
                 #'H',
                 #'+',
                 #'x',
                 #'D',
                 #'d',
                 #'|',
                 #'_'
                 ]
    linestyle = [
        '-',
        '--',
        '-.',
        ':',
        # 'None', draw nothing, 相当于画点
        # ' ', draw nothing, 相当于画点
        # '', draw nothing, 相当于画点
    ]

    if (index is None) or (index >= len(linestyle)):
        index = np.random.randint(0, len(linestyle))

    return linestyle[index]


def _get_plt_markers(index=None):
    markers = matplotlib.markers.MarkerStyle.markers
    markers = list(markers.keys())
    markers_cp = markers.copy()
    for mark in markers_cp:
        if type(mark) in (int,):
            markers.remove(mark)
        elif mark is None:
            markers.remove(mark)
    markers.sort()

    if (index is None) or (index >= len(markers)):
        index = np.random.randint(0, len(markers))

    return markers[index]


def _get_plt_style(index=None):
    def _test():
        y = np.arange(1, 5)
        for i in range(0, 20):
            color = _get_plt_color()
            linestyle = _get_plt_linestyle()
            marker = _get_plt_markers()

            plt.plot(y + i, linewidth=1, color=color, linestyle=linestyle, marker=marker)
            y_pt = y + i
            plt.text(3, y_pt[-1], color, color=color, fontsize=16)
            print('{}, {}'.format(i, color))
        plt.show()

    color = _get_plt_color(index)
    linestyle = _get_plt_linestyle(index)
    marker = _get_plt_markers(index)

    #_test()

    return color, linestyle, marker


def plt_roc(fprs, tprs, show_labels, interpret_label='', save_path=None):
    plt.figure(figsize=(10, 10))

    for i, (fpr, tpr, show_label) in enumerate(zip(fprs, tprs, show_labels)):
        color, linestyle, marker = _get_plt_style(i)

        plt.plot(fpr, tpr, color=color, linewidth=1, linestyle=linestyle, marker=marker, label=show_label)

    plt.xlabel(r'$\rm{fpr}$', fontsize=16)
    plt.ylabel(r'$\rm{tpr}$', fontsize=16)
    plt.title(r'ROC', fontsize=16)
    plt.text(np.max(fprs)*0.6, np.max(tprs)*1.06, r'{}'.format(interpret_label), fontsize=10)
    plt.legend(loc="lower right")  # 显示图例
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_embedding(data, label, title, save_path=None):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    num_classes = len(set(label))

    # fig = plt.figure()
    # ax = plt.subplot(111)
    # fontsize = min(len(data)/1200, 16)
    fontsize = 16
    print('[plot_embedding]: fontsize={}'.format(fontsize))
    for i in range(data.shape[0]):
        label_name = str(label[i])
        # color = plt.cm.Set(label[i] / num_classes)  # TODO: 有的时候会出错，暂不明什么原因。
        color = _get_plt_color(label[i])
        plt.text(data[i, 0], data[i, 1], label_name, color=color,
                 fontdict={'weight': 'bold', 'size': fontsize})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    # plt.legend()  # 显示图例
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def dim_reduct(features, rslt_dim=2):
    tsne = TSNE(n_components=rslt_dim, init='pca', random_state=1000, verbose=True)
    result = tsne.fit_transform(features)

    return result


def get_dataset(data_path, shuffle=True, validation_ratio=0.0):
    '''
data_path dir style:
data
├── folder1
│   ├── 00063.jpg
│   ├── 00068.jpg
└── folder2
    ├── 00070.jpg
    ├── 00072.jpg

    :param data_path:
    :return:
    '''
    raise Exception('已废弃，迁移至 tensorflow_stu/datasets/dataset.py中的load_dataset函数。')

    images_path = []
    images_label = []
    images = os.listdir(data_path)
    images.sort()
    if DEBUG:
        images = images[0:2]
    for i, image in enumerate(images):
        imgs = os.listdir(os.path.join(data_path, image))
        images_path.extend([os.path.join(data_path, image, img) for img in imgs])
        images_label.extend([i] * len(imgs))

    if shuffle:
        images = np.array([images_path, images_label]).transpose()
        np.random.shuffle(images)
        images_path = images[:, 0]
        images_label = images[:, 1].astype(np.int32)

    if DEBUG and False:
        images_path = images_path[0:500]
        images_label = images_label[0:500]

    if validation_ratio > 0.0:
        if not shuffle:
            raise Exception('When there is a validation set split requirement, shuffle must be True.')
        validation_size = int(len(images_path) * validation_ratio)
        validation_images_path = images_path[0:validation_size]
        validation_images_label = images_label[0:validation_size]

        train_images_path = images_path[validation_size:]
        train_images_label = images_label[validation_size:]
        return train_images_path, train_images_label, validation_images_path, validation_images_label
    else:
        return images_path, images_label


if __name__ == '__main__':  # plt_roc
    from sklearn import svm, datasets
    from sklearn import cross_validation

    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    ##变为2分类
    X, y = X[y != 2], y[y != 2]

    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=.3, random_state=0)

    # Learn to predict each class against the other
    svm = svm.SVC(kernel='linear', probability=True, random_state=random_state)

    ###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
    y_score = svm.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    # plt_roc(fpr, tpr, roc_auc, save_path='./tmp.jpg')
    plt_roc([fpr, fpr], [tpr, tpr], [roc_auc, roc_auc], interpret_label='test', save_path='./tmp.jpg')




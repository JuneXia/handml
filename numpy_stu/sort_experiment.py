import numpy as np

if __name__ == "__main__":
    arr = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

    # 输出arr中第0小值(即最小值)的下标
    np.argpartition(arr, 0)[0]

    # 输出arr中第1小值的下标
    np.argpartition(arr, 1)[1]

    # 输出arr中第len(arr)-1小值(即最大值)的下标
    np.argpartition(arr, len(arr) - 1)[len(arr) - 1]

    # 处处arr中第1大的值
    np.argpartition(arr, -1)[-1]

    # 处处arr中第2大的值
    np.argpartition(arr, -2)[-2]

    # 同时找到arr中第2和第4小值的下标，然后输出第2小值的下标
    np.argpartition(arr, [2, 4])[2]

    # 同时找到arr中第2和第4小值的下标，然后输出第4小值的下标
    np.argpartition(arr, [2, 4])[4]





import numpy as np
import matplotlib.pyplot as plt

var = 0.5
x = np.random.normal(loc=0.0, scale=var, size=100) * 0.5
y = np.random.normal(loc=0.0, scale=var, size=100) * 0.5

x2 = np.random.normal(loc=0.5, scale=var, size=100) * 0.5
y2 = np.random.normal(loc=0.5, scale=var, size=100) * 0.5

area = np.random.rand(10) * 100
fig = plt.figure()
plt.axis([-1.0, 1.0, -1.0, 1.0])  # xmin, xmax, ymin, ymax
# plt.axis([-1.0, 1.0, -4, 4])  # xmin, xmax, ymin, ymax
ax = plt.subplot()
ax.scatter(x, y, s=area, c='red', marker='o', alpha=0.6)
ax.scatter(x2, y2, s=area, c='green', marker='o', alpha=0.5)  # 改变颜色
plt.show()


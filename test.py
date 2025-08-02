import numpy as np
import matplotlib.pyplot as plt 
# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# fig, axs = plt.subplots(2,2,figsize=(5, 5), layout='constrained')
# fig.suptitle('多项式函数图像')
# axs[0,0].set_title('线性')
# axs[0,1].set_title('二次')
# axs[1,0].set_title('三次')
# axs[1,1].set_title('四次')
# x = np.linspace(-10, 10, 100)
# y1 = x
# y2 = x**2
# y3 = x**3
# y4 = x**4
# axs[0,0].plot(x, y1)
# axs[0,1].plot(x, y2)
# axs[1,0].plot(x, y3)
# axs[1,1].plot(x, y4)
# for ax in axs.flat:
#     ax.set(xlim=(-10, 10), ylim=(-10, 10), ylabel='some numbers')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.grid()
# plt.show()
np.random.seed(19680801)  # seed the random number generator.
data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
ax.scatter('a', 'b', c='c', s='d', data=data)
ax.set_xlabel('entry a')
ax.set_ylabel('entry b')
plt.show()
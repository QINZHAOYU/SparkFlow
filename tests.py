"""
Need the dot binary from the graphviz package (www.graphviz.org).
"""

import numpy as np
from dezero.core import Variable
from dezero.utils import plot_dot_graph

# -----------------------泰勒展开的正向计算图

# import math


# def my_sin(x, threshold=0.0001):
#     y = 0
#     for i in range(100000):
#         c = (-1) ** i / math.factorial(2 * i + 1)
#         t = c * x ** (2 * i + 1)
#         y = y + t
#         if abs(t.data) < threshold:
#             print(i)
#             break
#     return y


# x = Variable(np.array(np.pi / 4))
# y = my_sin(x)
# y.backward()
# print(y.data)
# gx = x.grad
# print(x.grad)

# # 只有正向计算图，因为my_sin()没有反向计算图,只是一个函数
# plot_dot_graph(y, verbose=False, to_file="./results/mysin_y.png")

# ----------Rosenbrock函数的反向梯度优化


# def rosenbrock(x0, x1):
#     y = 100 * (x1 - x0**2) ** 2 + (x0 - 1) ** 2
#     return y


# x0 = Variable(np.array(0.0))
# x1 = Variable(np.array(2.0))
# lr = 0.001
# iters = 1000

# for i in range(iters):
#     y = rosenbrock(x0, x1)

#     x0.cleargrad()
#     x1.cleargrad()
#     y.backward()

#     # 变量的梯度向量方向是其函数值增加最快的方向，所以逆梯度方向是函数值下降最快的方向。
#     # 梯度方向是增加最快方向可以通过方向导数来说明，grad_n = grad * n * cosθ
#     x0.data -= lr * x0.grad
#     x1.data -= lr * x1.grad
# print(x0, x1)


# -------------------- Sin函数的反向传播

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F

x = Variable(np.linspace(-7, 7, 200))
y = 7.8 * x
y.backward(create_graph=True)
if all(x.grad.data == 7.8):
    print("Correct!")
else:
    print("Incorrect!")

x.cleargrad()  # x 重用，但 grad 归零
y = F.sin(x)
y.backward(create_graph=True)
# 正向计算图
plot_dot_graph(y, verbose=True, to_file="./results/sin_y.png")

n = 3
for i in range(n):
    gx = x.grad
    gx.name = "gx" + str(i + 1)
    # 反向求导计算图，这里注意：
    # 1. 计算图没有正、反之分，都是从一个输出端开始追溯计算过程，只是这个过程是正是反
    # 2. 通过正向计算图反推反向计算图，但是一般反向过程更复杂，这是由于求导操作，
    #    即 backward() 函数复杂度增加导致的，然而这依然是可以逐层逐步追溯的。
    # 3. 注意每一层backward()函数的输入梯度。
    # 4. 高阶导数反向计算图可视化过程可能会有剪枝情况，通常分析对原始x的导数。
    plot_dot_graph(gx, False, f"./results/sin_gx{i+1}.png")

    x.cleargrad()  # x.grad 归零
    # (高阶求导不是复合求导，而是重新计算；不清除重用变量x.grad会导致一阶梯度加上二阶梯度，没有意义)
    gx.backward(create_graph=True)

gx = x.grad
gx.name = "gx" + str(n + 1)
plot_dot_graph(gx, False, f"./results/sin_gx{n+1}.png")


# ---------------------------- Tanh函数的反向传播

# import dezero.functions as F

# x = Variable(np.array(1.0))
# y = F.tanh(x)
# x.name = "x"
# y.name = "y"
# y.backward(create_graph=True)

# iters = 3

# for i in range(iters):
#     gx = x.grad
#     x.cleargrad()
#     gx.backward(create_graph=True)

# gx = x.grad
# gx.name = "gx" + str(iters + 1)
# plot_dot_graph(gx, verbose=False, to_file="./results/tanh_gx.png")

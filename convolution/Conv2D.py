from mxnet.gluon import nn
from corr2d import *

# 二维卷积层将输入和卷积核做互相关运算，并加上一个标量偏差来得到输出。
# 卷积层的模型参数包括了卷积核和标量偏差。

class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()
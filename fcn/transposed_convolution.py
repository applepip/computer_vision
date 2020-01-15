from mxnet import gluon, image, init, nd
from mxnet.gluon import data as gdata, loss as gloss, model_zoo, nn
import numpy as np
import sys
from utils import *


X = nd.arange(1, 17).reshape((1, 1, 4, 4))
K = nd.arange(1, 10).reshape((1, 1, 3, 3))
conv = nn.Conv2D(channels=1, kernel_size=3)
conv.initialize(init.Constant(K))

print(conv(X))
print(K)

# 下面我们将卷积核K改写成含有大量零元素的稀疏矩阵W，即权重矩阵。
# 权重矩阵的形状为(4, 16)，其中的非零元素来自卷积核K中的元素。将输
# # 入X逐行连结，得到长度为16的向量。然后将W与向量化的X做矩阵乘法，得
# # 到长度为4的向量。对其变形后，我们可以得到和上面卷积运算相同的结果。

W, k = nd.zeros((4, 16)), nd.zeros(11)
k[:3], k[4:7], k[8:] = K[0, 0, 0, :], K[0, 0, 1, :], K[0, 0, 2, :]
W[0, 0:11], W[1, 1:12], W[2, 4:15], W[3, 5:16] = k, k, k, k
nd.dot(W, X.reshape(16)).reshape((1, 1, 2, 2)), W

print(nd.dot(W, X.reshape(16)).reshape((1, 1, 2, 2)))
print(W)

# 卷积的反向传播函数的实现可以看作将函数输入乘以转置后的权重矩阵, 而转置卷积
# 层正好交换了卷积层的前向计算函数与反向传播函数：转置卷积层的这两个函数可以
# 看作将函数输入向量分别乘以权重的转置和权重。

# 例子:
# 构造一个卷积层conv，并设输入X的形状为(1, 3, 64, 64)。卷积输出Y的通道数增加
# 到10，但高和宽分别缩小了一半。

conv = nn.Conv2D(10, kernel_size=4, padding=1, strides=2)
conv.initialize()

X = nd.random.uniform(shape=(1, 3, 64, 64))
Y = conv(X)
print(Y.shape)

# 通过创建Conv2DTranspose实例来构造转置卷积层conv_trans。这里我们设conv_trans的
# 卷积核形状、填充以及步幅与conv中的相同，并设输出通道数为3。当输入为卷积层conv
# 的输出Y时，转置卷积层输出与卷积层输入的高和宽相同：转置卷积层将特征图的高和宽
# 分别放大了2倍。

conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize()
print(conv_trans(Y).shape)

# 下面我们使用一个基于ImageNet数据集预训练的ResNet-18模型来抽取图像特征, 并将该网
# 络实例记为pretrained_net。可以看到，该模型成员变量features的最后两层分别是全局最
# 大池化层GlobalAvgPool2D和样本变平层Flatten，而output模块包含了输出用的全连接层。

pretrained_net = model_zoo.vision.resnet18_v2(pretrained=True)
print(pretrained_net.features[-4:])
print(pretrained_net.output)

# 下面我们创建全卷积网络实例net。它复制了pretrained_net实例成员变量features里除去最
# 后两层的所有层以及预训练得到的模型参数。

net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)

# 给定高和宽分别为320和480的输入，net的前向计算将输入的高和宽减小至原来的 1/32 ，即10和15。

X = nd.random.uniform(shape=(1, 3, 320, 480))
print(net(X).shape)

# 接下来，我们通过 1×1 卷积层将输出通道数变换为Pascal VOC2012数据集的类别个数21。
# 最后，我们需要将特征图的高和宽放大32倍，从而变回输入图像的高和宽。

# 如果步幅为 s 、填充为 s/2 （假设 s/2 为整数）、卷积核的高和宽为 2s ，
# 转置卷积核将输入的高和宽分别放大 s 倍。

num_classes = 21
net.add(nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(num_classes, kernel_size=64, padding=16,
                           strides=32))

# 初始化转置卷积层

# 转置卷积层可以放大特征图。在图像处理中，我们有时需要将图像放大，即上采样（upsample。
# 上采样的方法有很多，常用的有双线性插值。

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return nd.array(weight)

# 实验一下用转置卷积层实现的双线性插值的上采样。构造一个将输入的高和宽放大2倍的转置卷积层，
# 并将其卷积核用bilinear_kernel函数初始化。

conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize(init.Constant(bilinear_kernel(3, 3, 4)))

# 读取图像X，将上采样的结果记作Y。为了打印图像，我们需要调整通道维的位置。

img = image.imread('img/catdog.jpg')
X = img.astype('float32').transpose((2, 0, 1)).expand_dims(axis=0) / 255
Y = conv_trans(X)
out_img = Y[0].transpose((1, 2, 0))

set_figsize()
print('input image shape:', img.shape)
plt.imshow(img.asnumpy())
print('output image shape:', out_img.shape)
plt.imshow(out_img.asnumpy())
plt.show()


crop_size, batch_size, colormap2label = (320, 480), 32, nd.zeros(256**3)
for i, cm in enumerate(VOC_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
voc_dir = 'data'

num_workers = 0 if sys.platform.startswith('win32') else 4
train_iter = gdata.DataLoader(
    VOCSegDataset(True, crop_size, voc_dir, colormap2label), batch_size,
    shuffle=True, last_batch='discard', num_workers=num_workers)
test_iter = gdata.DataLoader(
    VOCSegDataset(False, crop_size, voc_dir, colormap2label), batch_size,
    last_batch='discard', num_workers=num_workers)


ctx = try_all_gpus()
loss = gloss.SoftmaxCrossEntropyLoss(axis=1)
net.collect_params().reset_ctx(ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1,
                                                      'wd': 1e-3})
train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=5)

def predict(img):
    X = test_iter._dataset.normalize_image(img)
    X = X.transpose((2, 0, 1)).expand_dims(axis=0)
    pred = nd.argmax(net(X.as_in_context(ctx[0])), axis=1)
    return pred.reshape((pred.shape[1], pred.shape[2]))


def label2image(pred):
    colormap = nd.array(VOC_COLORMAP, ctx=ctx[0], dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]

test_images, test_labels = read_voc_images(is_train=False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 480, 320)
    X = image.fixed_crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X, pred, image.fixed_crop(test_labels[i], *crop_rect)]
show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n)
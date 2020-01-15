from mxnet import autograd, gluon, image, init, nd
from mxnet.gluon import model_zoo, nn
import time

from utils import *

content_img = image.imread('img/rainier.jpg')
plt.imshow(content_img.asnumpy())

plt.show()

style_img = image.imread('img/autumn_oak.jpg')
plt.imshow(style_img.asnumpy())

plt.show()

'''
预处理和后处理图像
'''

# 预处理函数preprocess对输入图像在RGB三个通道分别做标准化，并将结果变换成卷积神经网络接受的输入格式。
# 后处理函数postprocess则将输出图像中的像素值还原回标准化之前的值。
# 由于图像打印函数要求每个像素的浮点数值在0到1之间，我们使用clip函数对小于0和大于1的值分别取0和1。

rgb_mean = nd.array([0.485, 0.456, 0.406])
rgb_std = nd.array([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32') / 255 - rgb_mean) / rgb_std
    return img.transpose((2, 0, 1)).expand_dims(axis=0)

def postprocess(img):
    img = img[0].as_in_context(rgb_std.context)
    return (img.transpose((1, 2, 0)) * rgb_std + rgb_mean).clip(0, 1)

'''
抽取特征
'''

pretrained_net = model_zoo.vision.vgg19(pretrained=True)

# 越靠近输入层的输出越容易抽取图像的细节信息，反之则越容易抽取图像的全局信息。
# 为了避免合成图像过多保留内容图像的细节，我们选择VGG较靠近输出的层，也称内容层
# ，来输出图像的内容特征。我们还从VGG中选择不同层的输出来匹配局部和全局的样式,
# 这些层也叫样式层。这些层的索引可以通过打印pretrained_net实例来获取。

style_layers, content_layers = [0, 5, 10, 19, 28], [25]

# 在抽取特征时，我们只需要用到VGG从输入层到最靠近输出层的内容层或样式层之间的所有层。
# 下面构建一个新的网络net，它只保留需要用到的VGG的所有层。我们将使用net来抽取特征。

net = nn.Sequential()
for i in range(max(content_layers + style_layers) + 1):
    net.add(pretrained_net.features[i])

# 给定输入X，如果简单调用前向计算net(X)，只能获得最后一层的输出。由于我们还需要中间层
# 的输出，因此这里我们逐层计算，并保留内容层和样式层的输出。

def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

# 下面定义两个函数，其中get_contents函数对内容图像抽取内容特征，而get_styles函数则
# 对样式图像抽取样式特征。因为在训练时无须改变预训练的VGG的模型参数，所以我们可以
# 在训练开始之前就提取出内容图像的内容特征，以及样式图像的样式特征。由于合成图像
# 是样式迁移所需迭代的模型参数，我们只能在训练过程中通过调用extract_features函数来
# 抽取合成图像的内容特征和样式特征。

def get_contents(image_shape, ctx):
    content_X = preprocess(content_img, image_shape).copyto(ctx)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, ctx):
    style_X = preprocess(style_img, image_shape).copyto(ctx)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y

'''
内容损失
'''
# 与线性回归中的损失函数类似，内容损失通过平方误差函数衡量合成图像与内容图像在内容特征上
# 的差异。平方误差函数的两个输入均为extract_features函数计算所得到的内容层的输出。

def content_loss(Y_hat, Y):
    return (Y_hat - Y).square().mean()

'''
样式损失
'''
# 样式损失也一样通过平方误差函数衡量合成图像与样式图像在样式上的差异。为了表达样式层输出的
# 样式，我们先通过extract_features函数计算样式层的输出。

def gram(X):
    num_channels, n = X.shape[1], X.size // X.shape[1]
    X = X.reshape((num_channels, n))
    return nd.dot(X, X.T) / (num_channels * n)

# 样式损失的平方误差函数的两个格拉姆矩阵输入分别基于合成图像与样式图像的样式层输出。这里
# 假设基于样式图像的格拉姆矩阵gram_Y已经预先计算好了。

def style_loss(Y_hat, gram_Y):
    return (gram(Y_hat) - gram_Y).square().mean()

'''
总变差损失
'''
# 我们学到的合成图像里面有大量高频噪点，即有特别亮或者特别暗的颗粒像素。一种常用的降噪
# 方法是总变差降噪（total variation denoising）。降低总变差损失能够尽可能使邻近的像素值相似。

def tv_loss(Y_hat):
    return 0.5 * ((Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).abs().mean() +
                  (Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).abs().mean())

'''
损失函数
'''
# 样式迁移的损失函数即内容损失、样式损失和总变差损失的加权和。通过调节这些权值超参数，我们可以
# 权衡合成图像在保留内容、迁移样式以及降噪三方面的相对重要性。

content_weight, style_weight, tv_weight = 1, 1e3, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 分别计算内容损失、样式损失和总变差损失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    l = nd.add_n(*styles_l) + nd.add_n(*contents_l) + tv_l
    return contents_l, styles_l, tv_l, l

'''
 创建和初始化合成图像
'''
# 在样式迁移中，合成图像是唯一需要更新的变量。因此，我们可以定义一个简单的模型GeneratedImage，并将合
# 成图像视为模型参数。模型的前向计算只需返回模型参数即可。

class GeneratedImage(nn.Block):
    def __init__(self, img_shape, **kwargs):
        super(GeneratedImage, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=img_shape)

    def forward(self):
        return self.weight.data()

def get_inits(X, ctx, lr, styles_Y):
    gen_img = GeneratedImage(X.shape)
    gen_img.initialize(init.Constant(X), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(gen_img.collect_params(), 'adam',
                            {'learning_rate': lr})
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer

'''
训练
'''

def train(X, contents_Y, styles_Y, ctx, lr, max_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, ctx, lr, styles_Y)
    for i in range(max_epochs):
        start = time.time()
        with autograd.record():
            contents_Y_hat, styles_Y_hat = extract_features(
                X, content_layers, style_layers)
            contents_l, styles_l, tv_l, l = compute_loss(
                X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step(1)
        nd.waitall()
        if i % 50 == 0 and i != 0:
            print('epoch %3d, content loss %.2f, style loss %.2f, '
                  'TV loss %.2f, %.2f sec'
                  % (i, nd.add_n(*contents_l).asscalar(),
                     nd.add_n(*styles_l).asscalar(), tv_l.asscalar(),
                     time.time() - start))
        if i % lr_decay_epoch == 0 and i != 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
            print('change lr to %.1e' % trainer.learning_rate)
    return X

ctx, image_shape = try_gpu(), (225, 150)
net.collect_params().reset_ctx(ctx)
content_X, contents_Y = get_contents(image_shape, ctx)
_, styles_Y = get_styles(image_shape, ctx)
output = train(content_X, contents_Y, styles_Y, ctx, 0.01, 500, 200)

plt.imsave('img/neural-style-1.png', postprocess(output).asnumpy())

# 为了得到更加清晰的合成图像，下面我们在更大的 300×450 尺寸上训练。我们将
# 图的高和宽放大2倍，以初始化更大尺寸的合成图像。

image_shape = (450, 300)
_, content_Y = get_contents(image_shape, ctx)
_, style_Y = get_styles(image_shape, ctx)
X = preprocess(postprocess(output) * 255, image_shape)
output = train(X, content_Y, style_Y, ctx, 0.01, 300, 100)
plt.imsave('img/neural-style-2.png', postprocess(output).asnumpy())
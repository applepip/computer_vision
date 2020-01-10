from mxnet import autograd, contrib, gluon, image, init, nd
from mxnet.gluon import loss as gloss, nn
import time

'''
类别预测层
'''

# 设目标的类别个数为q。每个锚框的类别个数将是q+1，其中类别0表示锚框只包含背景。
# 在某个尺度下，设特征图的高和宽分别为h和w，如果以其中每个单元为中心生成a个锚
# 框，那么我们需要对h*w*a个锚框进行分类。考虑输出和输入同一空间坐标 (x,y)：输
# 出特征图上(x,y)坐标的通道里包含了以输入特征图(x,y)坐标为中心生成的所有锚框的
# 类别预测。因此输出通道数为a(q+1) ，其中索引为i(q+1)+j（ 0≤j≤q ）的通道代表
# 了索引为i的锚框有关类别索引为j的预测。
def cls_predictor(num_anchors, num_classes):
    # 输出通道数：num_anchors * (num_classes + 1)
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3,
                     padding=1)

'''
边界框预测层
'''

# 唯一不同的是，这里需要为每个锚框预测4个偏移量，而不是q+1个类别。

def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)

'''
连结多尺度的预测
'''

# 我们对同一批量数据构造两个不同尺度下的特征图Y1和Y2，其中Y2相对于Y1来说高和宽分别
# 减半。以类别预测为例，假设以Y1和Y2特征图中每个单元生成的锚框个数分别是5和3，当目
# 标类别个数为10时，类别预测输出的通道数分别为 5×(10+1)=55 和 3×(10+1)=33 。预测
# 输出的格式为(批量大小, 通道数, 高, 宽)。可以看到，除了批量大小外，其他维度大小均
# 不一样。我们需要将它们变形成统一的格式并将多尺度的预测连结，从而让后续计算更简单。

def forward(x, block):
    block.initialize()
    return block(x)

Y1 = forward(nd.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
Y2 = forward(nd.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
# 预测输出的格式为(批量大小, 通道数, 高, 宽)
print((Y1.shape, Y2.shape))

def flatten_pred(pred):
    return pred.transpose((0, 2, 3, 1)).flatten()

def concat_preds(preds):
    return nd.concat(*[flatten_pred(p) for p in preds], dim=1)

print(concat_preds([Y1, Y2]).shape)

'''
高和宽减半块
'''

# 为了在多尺度检测目标，下面定义高和宽减半块down_sample_blk。它串联了两个填充为1的 3×3 卷
# 积层和步幅为2的 2×2 最大池化层。我们知道，填充为1的 3×3 卷积层不改变特征图的形状，而后
# 面的池化层则直接将特征图的高和宽减半。由于 1×2+(3−1)+(3−1)=6 ，输出特征图中每个单元在输
# 入特征图上的感受野形状为 6×6 。可以看出，高和宽减半块使输出特征图中每个单元的感受野变得
# 更广阔。

def down_sample_blk(num_channels):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
            nn.BatchNorm(in_channels=num_channels),
            nn.Activation('relu'))
    blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
            nn.BatchNorm(in_channels=num_channels),
            nn.Activation('relu'))
    blk.add(nn.MaxPool2D(2))
    return blk

print(forward(nd.zeros((2, 3, 20, 20)), down_sample_blk(10)).shape)

'''
基础网络块
'''

# 基础网络块用来从原始图像中抽取特征。为了计算简洁，我们在这里构造一个小的基础网络。
# 该网络串联3个高和宽减半块，并逐步将通道数翻倍。当输入的原始图像的形状为256×256时，
# 基础网络块输出的特征图的形状为32×32 。

def base_net():
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk

print(forward(nd.zeros((2, 3, 256, 256)), base_net()).shape)

'''
完整的模型
'''

# 单发多框检测模型一共包含5个模块，每个模块输出的特征图既用来生成锚框，又用来预测这
# 些锚框的类别和偏移量。第一模块为基础网络块，第二模块至第四模块为高和宽减半块，第五
# 模块使用全局最大池化层将高和宽降到1。

def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk

# 接下来，我们定义每个模块如何进行前向计算。跟之前介绍的卷积神经网络不同，这里
# 不仅返回卷积计算输出的特征图Y，还返回根据Y生成的当前尺度的锚框，以及基于Y预
# 测的锚框类别和偏移量。

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = contrib.ndarray.MultiBoxPrior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

print(ratios)
print(num_anchors)

# 定义出完整的模型TinySSD
class TinySSD(nn.Block):

    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            # 即赋值语句self.blk_i = get_blk(i)
            setattr(self, 'blk_%d' % i, get_blk(i))
            setattr(self, 'cls_%d' % i, cls_predictor(num_anchors,
                                                      num_classes))
            setattr(self, 'bbox_%d' % i, bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self, 'blk_%d' % i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, 'blk_%d' % i), sizes[i], ratios[i],
                getattr(self, 'cls_%d' % i), getattr(self, 'bbox_%d' % i))
        # reshape函数中的0表示保持批量大小不变
        return (nd.concat(*anchors, dim=1),
                concat_preds(cls_preds).reshape(
                    (0, -1, self.num_classes + 1)), concat_preds(bbox_preds))

net = TinySSD(num_classes=1)
net.initialize()
X = nd.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)



from utils import *

from mxnet import contrib, gluon, image, nd
import numpy as np
np.set_printoptions(2)

'''
生成多个锚框
'''
# 假设输入图像高为h，宽为w。我们分别以图像的每个像素为中心生成不同形状的锚框。
img = image.imread('img/catdog.jpg').asnumpy()
h, w = img.shape[0:2]

print(img.shape)
print(h, w)


X = nd.random.uniform(shape=(1, 3, h, w))  # 构造输入数据(NDArray 1x3x561x728)
print(X.shape)

# MultiBoxPrior
#   该函数是用来在一张给定的特征图中，对于给定的大小比例和宽高比例，生成不同锚框，
#     函数原型为mxnet.ndarray.contrib.MultiBoxPrior(data=None, sizes=_Null, ratios=_Null, clip=_Null, steps=_Null, offsets=_Null, out=None, name=None, **kwargs)
#       data: 输入的特征图，一般形状为（批量大小，通道数、宽、高）
#       sizes: 要生成的锚框的大小比例列表
#       ratios: 要生成的锚框的宽高比例列表
#       clip: boolean 类型变量，设置是否要剪切超出边界的锚框，默认为0

Y = contrib.nd.MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape)  # Out[2]:(1, 2042040, 4)

# 我们看到，返回锚框变量y的形状为（批量大小，锚框个数，4）。将锚框变量y的形状变为（图像高，图像宽，
# 以相同像素为中心的锚框个数，4）后，我们就可以通过指定像素位置来获取所有以该像素为中心的锚框了

# 在目标检测时，我们首先生成多个锚框，然后为每个锚框预测类别以及偏移量，接着根据预测的
# 偏移量调整锚框位置从而得到预测边界框，最后筛选需要输出的预测边界框。

boxes = Y.reshape((h, w, 5, 4))

# 下面的例子里我们访问以（250，250）为中心的第一个锚框。它有4个元素，分别是锚框左上角
# 的x和y轴坐标和右下角的x和y轴坐标，其中x和y轴的坐标值分别已除以图像的宽和高，因此值
# 域均为0和1之间。

print(boxes[250, 250, 0, :])

# 描绘图像中以某个像素为中心的所有锚框,show_bboxes函数在图像上画出多个边界框。

def show_bboxes(axes, bboxes, labels=None, colors=None):
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])

    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.asnumpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

set_figsize()
bbox_scale = nd.array((w, h, w, h))
fig = plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
plt.show()



ground_truth = nd.array([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])

fig = plt.imshow(img)

show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
plt.show()


'''
 标注训练集的锚框
'''

# 通过左上角和右下角的坐标构造了5个需要标注的锚框，分别记为 A0,…,A4 （程序中索引从0开始）
anchors = nd.array([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
plt.show()

# 通过contrib.nd模块中的MultiBoxTarget函数来为锚框标注类别和偏移量。该函数将背景类别设为0，
# 并令从零开始的目标类别的整数索引自加1（1为狗，2为猫）。我们通过expand_dims函数为锚框和真
# 实边界框添加样本维，并构造形状为(批量大小, 包括背景的类别个数, 锚框数)的任意预测结果。

# MultiBoxTarget
#   该函数是用来对生成锚框进行标记，函数原型为MultiBoxTarget(anchor=None, label=None, cls_pred=None,
#                                       overlap_threshold=_Null, ignore_label=_Null, negative_mining_ratio=_Null,
#                                       negative_mining_thresh=_Null, minimum_negative_samples=_Null,
#                                       variances=_Null, out=None, name=None, **kwargs)
#                anchor:	输入的锚框，一般是通过MultiBoxPrior生成，形状为（1，锚框总数，4）
#                label:	训练集的真实标签，一般形状为（批量大小，每张图片最多的真实锚框数，5），第二维中，如果给定
#                         图片没有这么多锚框，将会用-1填充空白，最后一维中的元素为 类别标签+四个坐标值（归一化）
#                cls_pred: 对于第一个输入参数，即输入的锚框，预测的类别分数，形状一般为（批量大小，预测的
#                          总类别数+1，锚框总数），这个输入的作用是为了负采样（negative_mining），如果不设
#                          置负采样，那么这个输入并不影响输出
#                negative_mining_ratio：设置负采样的比例大小
#                negative_mining_thresh：设置负采样的阈值
#                minimum_negative_samples：最少的负采样样本

labels = contrib.nd.MultiBoxTarget(anchors.expand_dims(axis=0),
                                   ground_truth.expand_dims(axis=0),
                                   nd.zeros((1, 3, 5)))

# 第三项表示为锚框标注的类别
print(labels[2]) # Out[8]:[[0. 1. 2. 0. 2.]]<NDArray 1x5 @cpu(0)>

# 我们根据锚框与真实边界框在图像中的位置来分析这些标注的类别。首先，在所有的“锚框—真实边界框”的配对中，
# 锚框 A4 与猫的真实边界框的交并比最大，因此锚框 A4 的类别标注为猫。不考虑锚框 A4 或猫的真实边界框，在剩
# 余的“锚框—真实边界框”的配对中，最大交并比的配对为锚框 A1 和狗的真实边界框，因此锚框 A1 的类别标注为
# 狗。接下来遍历未标注的剩余3个锚框：与锚框 A0 交并比最大的真实边界框的类别为狗，但交并比小于阈值（默认为0.5），
# 因此类别标注为背景；与锚框 A2 交并比最大的真实边界框的类别为猫，且交并比大于阈值，因此类别标注为猫；与
# 锚框 A3 交并比最大的真实边界框的类别为猫，但交并比小于阈值，因此类别标注为背景。

# 返回值的第二项为掩码（mask）变量，形状为(批量大小, 锚框个数的四倍)。
print(labels[1]) # Out[9]:[[0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 1. 1. 1. 1.]]<NDArray 1x20 @cpu(0)>
# 返回值的第二项为掩码（mask）变量，形状为(批量大小, 锚框个数的四倍)。掩码变量中的元素与每个锚框的4个偏移量
# 一一对应。 由于我们不关心对背景的检测，有关负类的偏移量不应影响目标函数。通过按元素乘法，掩码变量中的0可
# 以在计算目标函数之前过滤掉负类的偏移量。

# 返回的第一项是为每个锚框标注的四个偏移量，其中负类锚框的偏移量标注为0。
print(labels[0])

'''
输出预测边界框
'''
# 在模型预测阶段，我们先为图像生成多个锚框，并为这些锚框一一预测类别和偏移量。随后，我们根据锚框及其预测偏移量得到预测边界框。
# 当锚框数量较多时，同一个目标上可能会输出较多相似的预测边界框。为了使结果更加简洁，我们可以移除相似的预测边界框。常用的方法叫
# 作非极大值抑制（non-maximum suppression，NMS）。

# 我们来描述一下非极大值抑制的工作原理。对于一个预测边界框B，模型会计算各个类别的预测概率。
# 设其中最大的预测概率为p，该概率所对应的类别即B的预测类别。我们也将p称为预测边界框B的置信度。
# 在同一图像上，我们将预测类别非背景的预测边界框按置信度从高到低排序，得到列表L。从L中选取置
# 信度最高的预测边界框B1作为基准，将所有与B1的交并比大于某阈值的非基准预测边界框从L中移除。
# 这里的阈值是预先设定的超参数。此时， L保留了置信度最高的预测边界框并移除了与其相似的其他预
# 测边界框。接下来，从L中选取置信度第二高的预测边界框B2作为基准，将所有与B2的交并比大于某阈值
# 的非基准预测边界框从L中移除。重复这一过程，直到L中所有的预测边界框都曾作为基准。此时L中任意
# 一对预测边界框的交并比都小于阈值。最终，输出列表L中的所有预测边界框。

# 先构造4个锚框,简单起见，我们假设预测偏移量全是0：预测边界框即锚框。最后，我们构造每个类别的预测概率。

anchors = nd.array([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                    [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])

offset_preds = nd.array([0] * anchors.size)

# 置信度
cls_probs = nd.array([[0] * 4,  # 背景的预测概率
                      [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                      [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率

# 在图像上打印预测边界框和它们的置信度。

fig = plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
plt.show()

# 我们使用contrib.nd模块的MultiBoxDetection函数来执行非极大值抑制并设阈值为0.5。
output = contrib.ndarray.MultiBoxDetection(
    cls_probs.expand_dims(axis=0), offset_preds.expand_dims(axis=0),
    anchors.expand_dims(axis=0), nms_threshold=0.5)

# 返回的结果的形状为(批量大小, 锚框个数, 6)。其中每一行的6个元素代表同一个预测边界框的输出信息。
# 第一个元素是索引从0开始计数的预测类别（0为狗，1为猫），其中-1表示背景或在非极大值抑制中被移除。
# 第二个元素是预测边界框的置信度。剩余的4个元素分别是预测边界框左上角的 x 和 y 轴坐标以及右下角
# 的 x 和 y 轴坐标（值域在0到1之间）。

# 我们移除掉类别为-1的预测边界框，并可视化非极大值抑制保留的结果。

fig = plt.imshow(img)
for i in output[0].asnumpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [nd.array(i[2:]) * bbox_scale], label)
plt.show()
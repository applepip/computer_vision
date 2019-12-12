from utils import *
from mxnet import contrib, image, nd

img = image.imread('img/catdog.jpg')
h, w = img.shape[0:2]
print(h, w)

# 我们将卷积神经网络的二维数组输出称为特征图。 通过定义特征图的形状来确定任
# 一图像上均匀采样的锚框中心。

set_figsize()

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

# 当特征图的宽和高分别设为fmap_w和fmap_h时，该函数将在任一图像上均匀采样fmap_h
# 行fmap_w列个像素，并分别以它们为中心生成大小为s（假设列表s长度为1）的不同宽高
# 比（ratios）的锚框。

def display_anchors(fmap_w, fmap_h, s):
    fmap = nd.zeros((1, 10, fmap_w, fmap_h))
    anchors = contrib.nd.MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = nd.array((w, h, w, h))
    show_bboxes(plt.imshow(img.asnumpy()).axes,
               anchors[0] * bbox_scale)

# 设锚框大小为0.15，特征图的高和宽分别为4
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
plt.show()

# 我们将特征图的高和宽分别减半，并用更大的锚框检测更大的目标。当锚框大小设0.4时，有
# 些锚框的区域有重合。

display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
plt.show()

# 最后，我们将特征图的高和宽进一步减半至1，并将锚框大小增至0.8。
# 此时锚框中心即图像中心。

display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
plt.show()

# 既然我们已在多个尺度上生成了不同大小的锚框，相应地，我们需要在不同尺度下检测不同大小的目标。
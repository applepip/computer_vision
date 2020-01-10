from mxnet import gluon, image, nd
import os
import sys
from utils import *

# 将输入图像和标签全部读进内存
def read_voc_images(root='data', is_train=True):
    txt_fname = '%s/ImageSets/Segmentation/%s' % (
        root, 'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in enumerate(images):
        features[i] = image.imread('%s/JPEGImages/%s.jpg' % (root, fname))
        labels[i] = image.imread(
            '%s/SegmentationClass/%s.png' % (root, fname))
    return features, labels

train_features, train_labels = read_voc_images()

n = 5
imgs = train_features[0:n] + train_labels[0:n]
show_images(imgs, 2, n)
plt.show()

# 我们列出标签中每个RGB颜色的值及其标注的类别。
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# 有了上面定义的两个常量以后，我们可以很容易地查找标签中每个像素的类别索引。

colormap2label = nd.zeros(256 ** 3)
for i, colormap in enumerate(VOC_COLORMAP):
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

# 本函数已保存在d2lzh包中方便以后使用
def voc_label_indices(colormap, colormap2label):
    colormap = colormap.astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

y = voc_label_indices(train_labels[0], colormap2label)
print(y[105:115, 130:140], VOC_CLASSES[1])

# 语义分割数据集类
class VOCSegDataset(gdata.Dataset):
    def __init__(self, is_train, crop_size, voc_dir, colormap2label):
        self.rgb_mean = nd.array([0.485, 0.456, 0.406])
        self.rgb_std = nd.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(root=voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = colormap2label
        print('read ' + str(len(self.features)) + ' examples')

    # 对输入图像的RGB三个通道的值分别做标准化
    def normalize_image(self, img):
        return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

    # 由于数据集中有些图像的尺寸可能小于随机裁剪所指定的输
    # 出尺寸，这些样本需要通过自定义的filter函数所移除。
    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[0] >= self.crop_size[0] and
            img.shape[1] >= self.crop_size[1])]

    # 任意访问数据集中索引为idx的输入图像及其每个像素的类别索引。
    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature.transpose((2, 0, 1)),
                voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

# 通过自定义的VOCSegDataset类来分别创建训练集和测试集的实例。
# # 假设我们指定随机裁剪的输出图像的形状为320×480。
# # 下面我们可以查看训练集和测试集所保留的样本个数。

crop_size = (320, 480)
voc_train = VOCSegDataset(True, crop_size, 'data', colormap2label)
voc_test = VOCSegDataset(False, crop_size, 'data', colormap2label)

print(len(voc_train))
print(len(voc_test))

# 设批量大小为64，分别定义训练集和测试集的迭代器。

batch_size = 64
num_workers = 0 if sys.platform.startswith('win32') else 4
train_iter = gdata.DataLoader(voc_train, batch_size, shuffle=True,
                              last_batch='discard', num_workers=num_workers)
test_iter = gdata.DataLoader(voc_test, batch_size, last_batch='discard',
                             num_workers=num_workers)

# 打印第一个小批量的形状。不同于图像分类和目标识别，这里的标签是一个三维数组。
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
# 我们使用ROIPooling函数来演示兴趣区域池化层的计算。假设卷
# 积神经网络抽取的特征X的高和宽均为4且只有单通道。

from mxnet import nd

X = nd.arange(16).reshape((1, 1, 4, 4))

print(X)

# 假设图像的高和宽均为40像素。再假设选择性搜索在图像上生成了两
# 个提议区域：每个区域由5个元素表示，分别为区域目标类别、左上角
# 的 x 和 y 轴坐标以及右下角的 x 和 y 轴坐标。

 # [区域目标类别, 左上角x, 左上角y, 右下角x, 右下角y]

rois = nd.array([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])


 # 由于X的高和宽是图像的高和宽的 1/10 ，以上两个提议区域中的坐标先按spatial_scale自乘0.1
 # , 然后在X上分别标出兴趣区域X[:,:,0:3,0:3]和X[:,:,1:4,0:4]。最后对这两个兴趣区域分别划
 # 分子窗口网格并抽取高和宽为2的特征。

print(X[:,:,0:3,0:3])

print(X[:,:,1:4,0:4])

roipooling = nd.ROIPooling(X, rois, pooled_size=(2, 2), spatial_scale=0.1)

print(roipooling)
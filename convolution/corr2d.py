from mxnet import autograd, nd

# 虽然卷积层得名于卷积（convolution）运算，但我们通常在卷积层中使用更加直观的互相
# 关（cross-correlation）运算。

def corr2d(X, K):
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = nd.array([[0, 1], [2, 3]])
Y = corr2d(X, K)
print(Y)

# Y: [[ 19.  25.]
#    [ 37.  43.]]


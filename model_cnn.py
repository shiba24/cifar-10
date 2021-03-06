import chainer
import chainer.functions as F
import chainer.links as L
from chainer.utils import conv


class CifarCNN(chainer.Chain):

    """
    Single-GPU AlexNet without partition toward the channel axis.
    Number of units in each layer was arranged for cifar-10 example,
    because input image = 32 x 32 x 3.
    """

    insize = 32

    def __init__(self):
        super(CifarCNN, self).__init__(
            conv1=L.Convolution2D(3,  96, 3, stride=2),
            conv2=L.Convolution2D(96, 256,  3, pad=1),
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  2, pad=1),
            conv5=L.Convolution2D(384, 256,  2, pad=1),
            fc6=L.Linear(2304, 2304),
            fc7=L.Linear(2304, 512),
            fc8=L.Linear(512, 10),
            )
        self.train = True

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=1)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self.fc8(h)

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss


class CifarCNN_2(chainer.Chain):

    """
    Single-GPU AlexNet without partition toward the channel axis.
    Number of units in each layer was arranged for cifar-10 example,
    because input image = 32 x 32 x 3.
    """
    insize = 32

    def __init__(self):
        super(CifarCNN_2, self).__init__(
            conv1=L.Convolution2D(3,  96, 3, pad=1),
            conv2=L.Convolution2D(96, 256,  3, pad=1),
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  3, pad=1),
            conv5=L.Convolution2D(384, 256,  3, pad=1),
            fc6=L.Linear(1024, 1024),
            fc7=L.Linear(1024, 128),
            fc8=L.Linear(128, 10),
            )
        self.train = True

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv1(x))), 2, stride=2)
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv2(h))), 2, stride=2)
        h = F.dropout(F.relu(self.conv3(h)), ratio=0.7, train=self.train)
        h = F.max_pooling_2d(F.relu(self.conv4(h)), 2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 2, stride=2, cover_all=True)
        h = F.dropout(F.relu(self.fc6(h)), ratio=0.7, train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), ratio=0.7, train=self.train)
        h = self.fc8(h)

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss


class CifarCNN_bn(chainer.Chain):
    """Single-GPU AlexNet with LRN layers replaced by BatchNormalization."""
    insize = 32

    def __init__(self):
        super(CifarCNN_bn, self).__init__(
            conv1=L.Convolution2D(3,  96, 3, pad=1),
            bn1=L.BatchNormalization(96),
            conv2=L.Convolution2D(96, 256,  3, pad=1),
            bn2=L.BatchNormalization(256),
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  3, pad=1),
            conv5=L.Convolution2D(384, 256,  3, pad=1),
            fc6=L.Linear(1024, 128),
            fc7=L.Linear(128, 10),
            )
        self.train = True

    def __call__(self, x, t):
        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), 2, stride=2)
        h = self.bn2(self.conv2(h), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), 2, stride=2)
        h = F.dropout(F.relu(self.conv3(h)), ratio=0.6, train=self.train)
        h = F.max_pooling_2d(F.relu(self.conv4(h)), 2, stride=2)
        h = F.average_pooling_2d(F.relu(self.conv5(h)), 2, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), ratio=0.6, train=self.train)
        h = self.fc7(h)

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss


class CifarCNN_bn_crop(chainer.Chain):
    """Single-GPU AlexNet with LRN layers replaced by BatchNormalization."""
    insize = 24

    def __init__(self):
        super(CifarCNN_bn_crop, self).__init__(
            conv1=L.Convolution2D(3,  96, 3, pad=1),
            bn1=L.BatchNormalization(96),
            conv2=L.Convolution2D(96, 256,  3, pad=1),
            bn2=L.BatchNormalization(256),
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  3, pad=1),
            conv5=L.Convolution2D(384, 256,  3, pad=1),
            fc6=L.Linear(256, 256),
            fc7=L.Linear(256, 10),
            )
        self.train = True

    def __call__(self, x, t, predict=False):
        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), 2, stride=2)
        h = self.bn2(self.conv2(h), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), 2, stride=2)
        h = F.dropout(F.relu(self.conv3(h)), ratio=0.6, train=self.train)
        h = F.max_pooling_2d(F.relu(self.conv4(h)), 2, stride=2)
        h = F.average_pooling_2d(F.relu(self.conv5(h)), 3, stride=1)
        h = F.dropout(F.relu(self.fc6(h)), ratio=0.6, train=self.train)
        h = self.fc7(h)

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        if predict:
            return h
        else:
            return self.loss

"""
        print('---------------------')
        print(len(x.data))
        print(len(x.data[0]))
        print(len(x.data[0][0]))
        print('---------------------')
"""

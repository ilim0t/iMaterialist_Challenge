#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


class Res_block(chainer.Chain):
    def __init__(self, out_channels, ksize, in_channels=None, init_stride=None, stride=1, pad=1):
        initializer = chainer.initializers.HeNormal()
        super(Res_block, self).__init__()
        with self.init_scope():
            # pre-activation
            self.bn1 = L.BatchNormalization(in_channels or out_channels)
            self.conv1 = L.Convolution2D(None, out_channels, ksize, init_stride or stride, pad, initialW=initializer)
            self.bn2 = L.BatchNormalization(out_channels)
            self.conv2 = L.Convolution2D(None, out_channels, ksize, stride, pad, initialW=initializer)
            self.bn3 = L.BatchNormalization(out_channels)

            self.xconv = L.Convolution2D(None, out_channels, 1, stride=2, initialW=initializer)

    def __call__(self, x, ratio):
        h = self.bn1(x)
        h = self.conv1(h)
        h = F.relu(self.bn2(h))
        h = F.dropout(h, ratio)  # Stochastic Depth
        h = self.conv2(h)
        h = self.bn3(h)  # 必要?

        if x.shape[2:] != h.shape[2:]:  # skipではないほうのデータの縦×横がこのblock中で小さくなっていた場合skipの方もそれに合わせて小さくする
            #x = F.average_pooling_2d(x, 1, 2)  # これでいいのか？
            x = self.xconv(x)
        if x.shape[1] != h.shape[1]:  # skipではない方のデータのチャンネル数がこのblock内で増えている場合skipの方もそれに合わせて増やす(zero-padding)
            xp = chainer.cuda.get_array_module(x.data)  # GPCが使える場合も想定
            p = chainer.Variable(xp.zeros((x.shape[0], h.shape[1] - x.shape[1], *x.shape[2:]), dtype=xp.float32))
            x = F.concat((x, p))
        return x + h


# Network definition
class ResNet(chainer.Chain):  # 18-layer
    def __init__(self, n_out, lossfunc=0):
        self.lossfunc = lossfunc
        initializer = chainer.initializers.HeNormal()
        super(ResNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 7, stride=2, pad=3)

            # Wide Residual Network
            self.block1_1 = Res_block(64, 3, pad=1)
            self.block1_2 = Res_block(64, 3, pad=1)

            self.block2_1 = Res_block(128, 3, 64, init_stride=2, pad=1)
            self.block2_2 = Res_block(128, 3, pad=1)

            self.block3_1 = Res_block(256, 3, 128, init_stride=2, pad=1)
            self.block3_2 = Res_block(256, 3, pad=1)

            self.block4_1 = Res_block(512, 3, 256, init_stride=2, pad=1)
            self.block4_2 = Res_block(512, 3, pad=1)

            #self.l1 = L.Linear(1000, initialW=initializer)
            self.l2 = L.Linear(n_out, initialW=initializer)

    def __call__(self, x):             # => 3   × 256
        h = self.conv1(x)              # => 64  ×  128
        h = F.max_pooling_2d(h, 3, 2)  # => 64  ×  64

        n = 1 / 2 / 8
        h = self.block1_1(h, 1*n)      # => 64  ×  64
        h = self.block1_2(h, 2*n)      # => 64  ×  64

        h = self.block2_1(h, 3*n)      # => 128 ×  32
        h = self.block2_2(h, 4*n)      # => 128 ×  32

        h = self.block3_1(h, 5*n)      # => 256 ×   16
        h = self.block3_2(h, 6*n)      # => 256 ×   16

        h = self.block4_1(h, 7*n)      # => 512 ×   8
        h = self.block4_2(h, 8*n)      # => 512 ×   8

        h = F.average_pooling_2d(h, h.shape[2])  # global average pooling

        # h = self.l1(h)
        return F.tanh(self.l2(h))

    def loss_func(self, x, t):
        y = self.__call__(x)
        loss = 0
        if self.lossfunc == 1 or self.lossfunc == -1:
            TP = F.sum((y + 1) * 0.5 * t, axis=1)
            FP = F.sum((y + 1) * 0.5 * (1 - t), axis=1)
            FN = F.sum((1 - y) * 0.5 * t, axis=1)
            # precision = TP / (TP + FP)
            # recall = TP / (TP + FN)
            loss += 1 - F.average(2 * TP / (2 * TP + FP + FN))  # F1 scoreを元に

        if self.lossfunc == 2:
            t_card = F.sum(t.astype("f"), axis=1)
            loss += F.average(F.sum(t * F.exp(- y), axis=1) * F.sum((1 - t) * F.exp(y), axis=1) /
                              (t_card * (t.shape[1] - t_card)))  # https://ieeexplore.ieee.org/document/1683770/ (3)式を変形
            """
            |Y_i|行列  = t_card = F.sum(t.astype("f"), axis=1)
            |Y[bar]_i|行列 = t.shape[1] - t_card
            
             Σ exp(-(C_k-C_l))
            =Σ exp(-C_k+C_l)
            =Σ exp(-C_k)*exp(C_l)
            =(exp(-C_k1), exp(-C_k2), ...) * (exp(C_l1), exp(C_l2), ...)
            =(exp(-x) dot [1 if x∈Y else 0]) * (exp(x)行列 dot [0 if x∈Y else 1])
            =F.sum(t * F.exp(- y), axis=1) * F.sum((1 - t) * F.exp(y), axis=1)
            
            ( TP  + FN ) * ( TN  + FP )
            (e^-1 + e^1) * (e^-1 + e^1)
            
            ∴loss = loss += F.average(F.sum(t * F.exp(- y), axis=1) * F.sum((1 - t) * F.exp(y), axis=1) / (t_card * (t.shape[1] - t_card)))
            """
        if not hasattr(self, 'n'):
            self.n = -15
            self.fnk = np.e
            self.fpk = np.e / (0.9 ** 13)
            self.tpk = np.e
        if self.lossfunc == 0 or self.lossfunc == -1:
            tpk = - np.log(self.tpk)
            fnk = np.log(self.fnk)
            tnk = - 1
            fpk = np.log(self.fpk)  # np.log(20 - 20 * self.a/100 + np.e)
            t_card = F.sum(t.astype("f"), axis=1)
            loss += F.average(
                F.sum(t * F.exp((y*(tpk - fnk) + tpk + fnk) / 2), axis=1) *
                F.sum((1 - t) * F.exp((y*(fpk - tnk) + fpk + tnk) / 2), axis=1)
                / (t_card * (t.shape[1] - t_card)))  # https://ieeexplore.ieee.org/document/1683770/ (3)式を変形 & FPに対し重み付け

        chainer.reporter.report({'loss': loss}, self)
        accuracy = self.accuracy(y.data, t)

        siggma = 5.836
        if accuracy[9].data > siggma:
            self.n += 1
            if self.n >= 20:
                self.n = 0
                # self.shift.n = 0
                self.fpk = min(self.fpk / 0.9, 40)
        elif accuracy[9].data < siggma:
            self.n -= 1
            if self.n <= -20:
                self.n = 0
                self.n = 0
                # self.shift.n = 0
                self.fpk = max(self.fpk * 0.9, 0.1)

        chainer.reporter.report({'fpk': self.fpk}, self)
        chainer.reporter.report({'acc': accuracy[0]}, self)  # dataひとつひとつのlabelが完全一致している確率
        chainer.reporter.report({'acc2': accuracy[1]}, self)  # すべてのbatchを通してlabelそれぞれの正解確率の平均
        # chainer.reporter.report({'acc_66': accuracy[2]}, self)  # 66番ラベルの正解率
        chainer.reporter.report({'precision': accuracy[5]}, self)
        chainer.reporter.report({'recall': accuracy[6]}, self)
        chainer.reporter.report({'f1': accuracy[3]}, self)
        chainer.reporter.report({'labelnum': accuracy[9]}, self)
        # chainer.reporter.report({'freq_err': accuracy[4]}, self)  # batchの中で最も多く間違って判断したlabel
        return loss

    def accuracy(self, y, t):
        y = chainer.cuda.to_cpu(y)
        t = chainer.cuda.to_cpu(t)

        y_binary = (y > 0).astype(int)
        accuracy = sum([1 if all(i) else 0 for i in (y_binary == t)]) / len(y)  # dataひとつひとつのlabelが完全一致している確率
        accuracy2 = np.sum((y_binary == t).astype(int)) / len(y) / len(y[0])  # すべてのbatchを通してlabelそれぞれの正解確率の平均
        acc_66 = np.sum((y_binary[:, 65] == t[:, 65]).astype(int)) / len(y)  # 66番ラベルの正解率
        frequent_error = np.sum((y_binary != t).astype(int), 0).argsort()[-1] + 1  # batchの中で最も多く間違って判断したlabel

        TP = np.sum(y_binary * t, axis=1)
        FP = np.sum(y_binary * (1 - t), axis=1)
        FN = np.sum((1 - y_binary) * t, axis=1)
        f1 = np.average(2 * TP / (2 * TP + FP + FN))
        return accuracy, accuracy2, acc_66, f1, frequent_error,\
               np.average(TP/(TP+FP)), np.average(TP/(TP+FN)), np.var(TP/(TP+FP)), np.var(TP/(TP+FN)), \
               F.average(F.sum(y_binary.astype(float), axis=1))

class Bottle_neck_block(chainer.Chain):
    def __init__(self, out_channels, ksize, in_channels=None, init_stride=None, stride=1, pad=1):
        initializer = chainer.initializers.HeNormal()
        middle_channels = int(out_channels / 2)
        super(Bottle_neck_block, self).__init__()
        with self.init_scope():
            # pre-activation & 参考: https://arxiv.org/pdf/1610.02915.pdf
            self.bn1 = L.BatchNormalization(in_channels or out_channels)
            self.conv1 = L.Convolution2D(None, middle_channels, 1, initialW=initializer)
            self.bn2 = L.BatchNormalization(middle_channels)
            self.conv2 = L.Convolution2D(None, middle_channels, ksize, init_stride or stride, pad, initialW=initializer)
            self.bn3 = L.BatchNormalization(middle_channels)
            self.conv3 = L.Convolution2D(None, out_channels, 1, initialW=initializer)
            self.bn4 = L.BatchNormalization(out_channels)

            self.xconv = L.Convolution2D(None, out_channels, 1, stride=2, initialW=initializer)

    def __call__(self, x, ratio):
        h = self.bn1(x)
        h = self.conv1(h)
        h = F.relu(self.bn2(h))
        h = self.conv2(h)
        h = F.relu(self.bn3(h))
        h = F.dropout(h, ratio)  # Stochastic Depth
        h = self.conv3(h)
        h = self.bn4(h)  # 必要?

        if x.shape[2:] != h.shape[2:]:  # skipではないほうのデータの縦×横がこのblock中で小さくなっていた場合skipの方もそれに合わせて小さくする
            #x = F.average_pooling_2d(x, 1, 2)  # これでいいのか？
            x = self.xconv(x)
        if x.shape[1] != h.shape[1]:  # skipではない方のデータのチャンネル数がこのblock内で増えている場合skipの方もそれに合わせて増やす(zero-padding)
            xp = chainer.cuda.get_array_module(x.data)  # GPUが使える場合も想定
            p = chainer.Variable(xp.zeros((x.shape[0], h.shape[1] - x.shape[1], *x.shape[2:]), dtype=xp.float32))
            x = F.concat((x, p))
        return x + h


class Bottle_neck_RES_net(ResNet):  # 18-layer
    def __init__(self, n_out, lossfunc=0):
        self.lossfunc = lossfunc
        initializer = chainer.initializers.HeNormal()
        super(ResNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 7, stride=2, pad=3)

            # Wide Residual Network
            self.block1_1 = Bottle_neck_block(64, 3, pad=1)
            self.block1_2 = Bottle_neck_block(64, 3, pad=1)

            self.block2_1 = Bottle_neck_block(128, 3, 64, init_stride=2, pad=1)
            self.block2_2 = Bottle_neck_block(128, 3, pad=1)

            self.block3_1 = Bottle_neck_block(256, 3, 128, init_stride=2, pad=1)
            self.block3_2 = Bottle_neck_block(256, 3, pad=1)

            self.block4_1 = Bottle_neck_block(512, 3, 256, init_stride=2, pad=1)
            self.block4_2 = Bottle_neck_block(512, 3, pad=1)

            #self.l1 = L.Linear(1000, initialW=initializer)
            self.l2 = L.Linear(n_out, initialW=initializer)


class ResNet_lite(ResNet):  # 14?-layer
    def __init__(self, n_out, lossfunc=0):
        self.lossfunc = lossfunc
        initializer = chainer.initializers.HeNormal()
        super(ResNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 7, stride=2, pad=3)

            # Wide Residual Network
            self.block1_1 = Res_block(64, 3, pad=1)
            self.block1_2 = Res_block(64, 3, pad=1)

            self.block2_1 = Res_block(128, 3, 64, init_stride=2, pad=1)
            self.block2_2 = Res_block(128, 3, pad=1)

            self.block3_1 = Res_block(256, 3, 128, init_stride=2, pad=1)
            self.block3_2 = Res_block(256, 3, pad=1)

            #self.l1 = L.Linear(1000, initialW=initializer)
            self.l2 = L.Linear(n_out, initialW=initializer)

    def __call__(self, x):             # => 3   × 128
        h = self.conv1(x)              # => 64  ×  64
        h = F.max_pooling_2d(h, 3, 2)  # => 64  ×  32

        n = 1 / 2 / 6
        h = self.block1_1(h, 1*n)      # => 64  ×  32
        h = self.block1_2(h, 2*n)      # => 64  ×  32

        h = self.block2_1(h, 3*n)      # => 128 ×  16
        h = self.block2_2(h, 4*n)      # => 128 ×  16

        h = self.block3_1(h, 5*n)      # => 256 ×   8
        h = self.block3_2(h, 6*n)      # => 256 ×   8

        h = F.average_pooling_2d(h, h.shape[2])  # global average pooling

        # h = self.l1(h)
        return F.tanh(self.l2(h))


class Bottle_neck_RES_net_lite(ResNet_lite):  # 18-layer
    def __init__(self, n_out, lossfunc=0):
        self.lossfunc = lossfunc
        initializer = chainer.initializers.HeNormal()
        super(ResNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 7, stride=2, pad=3)

            # Wide Residual Network
            self.block1_1 = Bottle_neck_block(64, 3, pad=1)
            self.block1_2 = Bottle_neck_block(64, 3, pad=1)

            self.block2_1 = Bottle_neck_block(128, 3, 64, init_stride=2, pad=1)
            self.block2_2 = Bottle_neck_block(128, 3, pad=1)

            self.block3_1 = Bottle_neck_block(256, 3, 128, init_stride=2, pad=1)
            self.block3_2 = Bottle_neck_block(256, 3, pad=1)

            #self.l1 = L.Linear(1000, initialW=initializer)
            self.l2 = L.Linear(n_out, initialW=initializer)

class RES_SPP_block(chainer.Chain):
    def __init__(self, out_channels, ksize, in_channels=None, init_stride=None, stride=1, pad=1):
        initializer = chainer.initializers.HeNormal()
        super(RES_SPP_block, self).__init__()
        with self.init_scope():
            self.bn1 = L.BatchNormalization(in_channels or out_channels)
            self.conv1 = L.Convolution2D(None, out_channels, ksize, init_stride or stride, pad, initialW=initializer)
            self.bn2 = L.BatchNormalization(out_channels)
            self.conv2 = L.Convolution2D(None, out_channels, ksize, stride, pad, initialW=initializer)

    def __call__(self, x, ratio):
        h = F.relu(self.bn1(x))
        h = self.conv1(h)
        h = F.relu(self.bn2(h))
        h = F.dropout(h, ratio)
        h = self.conv2(h)

        if x.shape[2:] != h.shape[2:]:  # skipではないほうのデータの縦×横がこのblock中で小さくなっていた場合skipの方もそれに合わせて小さくする
            x = F.average_pooling_2d(x, 1, 2)
        if x.shape[1] != h.shape[1]:  # skipではない方のデータのチャンネル数がこのblock内で増えている場合skipの方もそれに合わせて増やす
            xp = chainer.cuda.get_array_module(x.data)  # GPUが使える場合も想定
            p = chainer.Variable(xp.zeros((x.shape[0], h.shape[1] - x.shape[1], *x.shape[2:]), dtype=xp.float32))
            x = F.concat((x, p))
        return x + h


# Network definition
class RES_SPP_net(ResNet):
    def __init__(self, n_out, lossfunc=0):
        self.lossfunc = lossfunc
        initializer = chainer.initializers.HeNormal()
        super(ResNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 7, pad=2)

            self.block1_1 = RES_SPP_block(64, 3, init_stride=2, pad=1)
            self.block1_2 = RES_SPP_block(64, 3, pad=1)
            self.block1_3 = RES_SPP_block(64, 3, pad=1)

            self.block2_1 = RES_SPP_block(128, 3, 64, init_stride=2, pad=1)
            self.block2_2 = RES_SPP_block(128, 3, pad=1)
            self.block2_3 = RES_SPP_block(128, 3, pad=1)

            self.block3_1 = RES_SPP_block(256, 3, 128, init_stride=2, pad=1)
            self.block3_2 = RES_SPP_block(256, 3, pad=1)
            self.block3_3 = RES_SPP_block(256, 3, pad=1)

            self.l1 = L.Linear(2048, initialW=initializer)
            self.l2 = L.Linear(n_out, initialW=initializer)

    def __call__(self, x):
        h = self.conv1(x)
        F.max_pooling_2d(h, 2)

        h = self.block1_1(h, 0.2)
        h = self.block1_2(h, 0.2)
        h = self.block1_3(h, 0.2)

        h = self.block2_1(h, 0.3)
        h = self.block2_2(h, 0.3)
        h = self.block2_3(h, 0.3)

        h = self.block3_1(h, 0.4)
        h = self.block3_2(h, 0.4)
        h = self.block3_3(h, 0.4)
        # h = F.spatial_pyramid_pooling_2d(h, 3, F.average_pooling_2d)
        h = F.spatial_pyramid_pooling_2d(h, 4, F.MaxPooling2D)
        h = self.l1(h)
        return F.tanh(self.l2(h))


import matplotlib as mpl
import os


class Myblock(chainer.Chain):
    """
    畳み込み層
    """
    def __init__(self, out_channels, ksize, stride=1, pad=0):
        initializer = chainer.initializers.HeNormal()
        super(Myblock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, out_channels, ksize, stride, pad, initialW=initializer)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return F.relu(h)


class Mymodel(ResNet):
    def __init__(self, n_out, lossfunc):
        self.n = 1
        self.lossfunc = lossfunc
        self.accs = [[], [], [], [], [], []]
        super(ResNet, self).__init__()
        initializer = chainer.initializers.HeNormal()
        with self.init_scope():
            # self.block1 = Block(32, 5, pad=1)  # n_in = args.size (300)^2 * 3 = 270000
            # self.block2 = Block(64, 3, pad=1)
            # self.block3 = Block(128, 3, pad=1)
            # self.block4 = Block(256, 3, pad=1)
            # self.block5 = Block(128, 3, pad=1)

            self.block1 = Myblock(32, 3)  # n_in = args.size (300)^2 * 3 = 270000
            self.block2 = Myblock(64, 2)
            self.block3 = Myblock(128, 2)
            self.block4 = Myblock(256, 2)
            self.block5 = Myblock(256, 2)
            self.block6 = Myblock(128, 2)

            self.fc1 = L.Linear(512, initialW=initializer)
            self.fc2 = L.Linear(512, initialW=initializer)
            # ↓中身を調べている最中
            # self.bn_fc1 = L.BatchNormalization(512)
            self.fc3 = L.Linear(n_out)

    def __call__(self, x):
        h = self.block1(x)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2)
        h = self.block2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2)
        h = self.block3(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2)
        h = self.block4(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2)
        h = self.block5(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2)
        h = self.block6(h)
        h = F.relu(h)

        h = self.fc1(h)
        h = F.dropout(h, ratio=0.2)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.dropout(h, ratio=0.1)
        h = F.relu(h)
        return F.tanh(self.fc3(h))

    def plot_acc(self, accuracy):
        for i in range(len(self.accs)):
            self.accs[i].append(accuracy[i])
        self.plot(self.accs[:2] + self.accs[3:])
        self.n += 1

    def plot(self, x):
        for i, j in enumerate(x):
            name = ['loss', 'accuracy', 'acc_66', 'accuracy2', 'f1'][i]
            fig = mpl.pyplot.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(range(1, self.n + 1), j, linestyle="dashed")
            for o in [5, 20]:  # 平均移動線
                if self.n >= o:
                    average_line = [0] * (self.n - o + 1)
                    for k, l in enumerate(j):
                        for m in range(o):
                            if k - m >= 0 and k - m < len(average_line):
                                average_line[k - m] += l
                    ax.plot(range(o, self.n + 1), [y / o for y in average_line])

            if i == 0:
                ax.set_ylim(None, 1.6)
            elif i == 2 or i == 3:
                ax.set_ylim(0.7, None)
            # ax.set_xticks(range(1, self.n + 1))
            ax.set_xlabel('iter')
            # ax.set_yticks([i / 10 for i in range(1, 10)])
            ax.set_ylabel(name)

            # ax.legend(loc='best')
            # ax.set_title(name)

            # save as png
            if not os.path.isdir('progress'):
                os.mkdir('progress')
            mpl.pyplot.savefig('progress/' + name + '.png')
            mpl.pyplot.clf()


class Lite(ResNet):
    def __init__(self, n_out, lossfunc):
        self.lossfunc = lossfunc
        super(ResNet, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(4, 8, stride=4)
            self.bn = L.BatchNormalization(4)
            self.fc = L.Linear(n_out)

    def __call__(self, x):
        h = F.max_pooling_2d(x, 4)
        h = self.conv(h)
        h = self.bn(h)
        h = self.fc(h)
        return F.tanh(h)

class FineVGG(ResNet):
    def __init__(self, n_out, lossfunc=0):
        self.lossfunc = lossfunc
        initializer = chainer.initializers.HeNormal()
        super(ResNet, self).__init__()
        with self.init_scope():
            self.base = L.VGG16Layers()
            self.l1 = L.Linear(4096, initialW=initializer)
            self.l2 = L.Linear(4096, initialW=initializer)
            self.l3 = L.Linear(n_out, initialW=initializer)

    def __call__(self, x):
        h = self.base(x, layers=['pool5'])
        h = F.relu(self.l1(h['pool5']))
        h = F.relu(self.l2(h))
        return F.tanh(self.l3(h))

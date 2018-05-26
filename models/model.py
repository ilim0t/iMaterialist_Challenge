#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


class Block(chainer.Chain):
    def __init__(self, out_channels, ksize, in_channels=None, init_stride=None, stride=1, pad=1):
        initializer = chainer.initializers.HeNormal()
        super(Block, self).__init__()
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
            xp = chainer.cuda.get_array_module(x.data)  # gupが使える場合も想定
            p = chainer.Variable(xp.zeros((x.shape[0], h.shape[1] - x.shape[1], *x.shape[2:]), dtype=xp.float32))
            x = F.concat((x, p))
        return x + h


# Network definition
class ResNet(chainer.Chain):
    def __init__(self, n_out):
        initializer = chainer.initializers.HeNormal()
        super(ResNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 7, pad=2)

            self.block1_1 = Block(64, 3, init_stride=2, pad=1)
            self.block1_2 = Block(64, 3, pad=1)
            self.block1_3 = Block(64, 3, pad=1)

            self.block2_1 = Block(128, 3, 64, init_stride=2, pad=1)
            self.block2_2 = Block(128, 3, pad=1)
            self.block2_3 = Block(128, 3, pad=1)

            self.block3_1 = Block(256, 3, 128, init_stride=2, pad=1)
            self.block3_2 = Block(256, 3, pad=1)
            self.block3_3 = Block(256, 3, pad=1)

            self.block4_1 = Block(512, 3, 256, init_stride=2, pad=1)
            self.block4_2 = Block(512, 3, pad=1)
            self.block4_3 = Block(512, 3, pad=1)

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

        h = self.block4_1(h, 0.5)
        h = self.block4_2(h, 0.5)
        h = self.block4_3(h, 0.5)

        h = F.average_pooling_2d(h, 2)
        h = self.l1(h)
        return self.l2(h)

    def loss_func(self, x, t):
        y = self.__call__(x)

        TP = F.sum((y + 1) * 0.5 * t, axis=1)
        FP = F.sum((y + 1) * 0.5 * (1 - t), axis=1)
        FN = F.sum((1 - y) * 0.5 * t, axis=1)

        loss = 1 - F.average(2 * TP / (2 * TP + FP + FN))  # F1 scoreを元に

        # t_card = F.sum(t.astype("f"), axis=1)
        # loss += F.average(F.sum(t * F.exp(- y), axis=1) * F.sum((1 - t) * F.exp(y), axis=1) /
        #                   (t_card * (t.shape[1] - t_card)))  # https://ieeexplore.ieee.org/document/1683770/ (3)式を変形

        chainer.reporter.report({'loss': loss}, self)

        accuracy = self.accuracy(y.data, t)
        chainer.reporter.report({'acc': accuracy[0]}, self)  # dataひとつひとつのlabelが完全一致している確率
        chainer.reporter.report({'acc2': accuracy[1]}, self)  # すべてのbatchを通してlabelそれぞれの正解確率の平均
        #chainer.reporter.report({'acc_66': accuracy[2]}, self)  # 66番ラベルの正解率
        chainer.reporter.report({'f1': accuracy[3]}, self)
        #chainer.reporter.report({'freq_err': accuracy[4]}, self)  # batchの中で最も多く間違って判断したlabel
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
        return accuracy, accuracy2, acc_66, f1, frequent_error


#以下は過去のmodel


class Block2(chainer.Chain):
    """
    畳み込み層
    """
    def __init__(self, out_channels, ksize, stride=1, pad=0):
        initializer = chainer.initializers.HeNormal()
        super(Block2, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, out_channels, ksize, stride, pad, initialW=initializer)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return F.relu(h)


class Mymodel(chainer.Chain):
    def __init__(self, n_out):
        self.n = 1
        self.accs = [[], [], [], [], [], []]
        super(Mymodel, self).__init__()
        initializer = chainer.initializers.HeNormal()
        with self.init_scope():
            # self.block1 = Block(32, 5, pad=1)  # n_in = args.size (300)^2 * 3 = 270000
            # self.block2 = Block(64, 3, pad=1)
            # self.block3 = Block(128, 3, pad=1)
            # self.block4 = Block(256, 3, pad=1)
            # self.block5 = Block(128, 3, pad=1)

            self.block1 = Block2(32, 3)  # n_in = args.size (300)^2 * 3 = 270000
            self.block2 = Block2(64, 2)
            self.block3 = Block2(128, 2)
            self.block4 = Block2(256, 2)
            self.block5 = Block2(256, 2)
            self.block6 = Block2(128, 2)

            self.fc1 = L.Linear(512, initialW=initializer)
            self.fc2 = L.Linear(512, initialW=initializer)
            # ↓中身を調べている最中
            # self.bn_fc1 = L.BatchNormalization(512)
            self.fc3 = L.Linear(n_out)

    def loss_func(self, x, t):
        y = self.predict(x)
        t_card = F.sum(t.astype("f"), axis=1)

        TP = F.sum((y + 1) * 0.5 * t, axis=1)
        FP = F.sum((y + 1) * 0.5 * (1 - t), axis=1)
        FN = F.sum((1 - y) * 0.5 * t, axis=1)

        loss = 1 - F.average(2 * TP / (2 * TP + FP + FN))  # F1 scoreを元に
        # loss += F.average(F.sum(t * F.exp(- y), axis=1) * F.sum((1 - t) * F.exp(y), axis=1) /
        #                   (t_card * (t.shape[1] - t_card)))  # https://ieeexplore.ieee.org/document/1683770/ (3)式を変形

        chainer.reporter.report({'loss': loss}, self)
        accuracy = self.accuracy(y.data, t)
        chainer.reporter.report({'acc': accuracy[0]}, self)  # dataひとつひとつのlabelが完全一致している確率
        chainer.reporter.report({'freq_err': accuracy[1]}, self)  # batchの中で最も多く間違って判断したlabel
        chainer.reporter.report({'acc_66': accuracy[2]}, self)  # 66番ラベルの正解率
        chainer.reporter.report({'acc2': accuracy[3]}, self)  # すべてのbatchを通してlabelそれぞれの正解確率の平均
        chainer.reporter.report({'f1': accuracy[4]}, self)

        if t.shape[0] == 256:
            self.plot_acc([loss.data] + list(accuracy))
        return loss

    def accuracy(self, y, t):
        y = chainer.cuda.to_cpu(y)
        t = chainer.cuda.to_cpu(t)

        y_binary = (y > 0).astype(int)
        accuracy = sum([1 if all(i) else 0 for i in (y_binary == t)]) / len(y)  # dataひとつひとつのlabelが完全一致している確率
        frequent_error = np.sum((y_binary != t).astype(int), 0).argsort()[-1] + 1  # batchの中で最も多く間違って判断したlabel
        acc_66 = np.sum((y_binary[:, 65] == t[:, 65]).astype(int)) / len(y)  # 66番ラベルの正解率
        accuracy2 = np.sum((y_binary == t).astype(int)) / len(y) / len(y[0])  # すべてのbatchを通してlabelそれぞれの正解確率の平均

        TP = np.sum(y_binary * t, axis=1)
        FP = np.sum(y_binary * (1 - t), axis=1)
        FN = np.sum((1 - y_binary) * t, axis=1)
        f1 = np.average(2 * TP / (2 * TP + FP + FN))
        return accuracy, frequent_error, acc_66, accuracy2, f1

    def predict(self, x):
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

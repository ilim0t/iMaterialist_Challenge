#!/usr/bin/env python
# -*- coding: utf-8 -*-


import warnings

warnings.filterwarnings('ignore', category=FutureWarning, message="Conversion of the second")
warnings.filterwarnings('ignore', category=RuntimeWarning, message="overflow encountered in exp")
warnings.filterwarnings('ignore', category=RuntimeWarning, message="invalid value encountered in sqrt")
warnings.filterwarnings('ignore', category=RuntimeWarning, message="More than 20 figures have been opened.")

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions

import numpy as np
import argparse

import os
import json
from PIL import Image

import matplotlib as mpl
import platform

# for rendering graph on remote server.
# see: https://qiita.com/TomokIshii/items/3a26ee4453f535a69e9e
if 1 or platform.system() != "Darwin":
    mpl.use('Agg')
import matplotlib.pyplot as plt

from scipy.ndimage.interpolation import rotate
from scipy.misc import imresize

#import chainer.training.extensions.evaluator

import copy
import six
from chainer.dataset import convert
from chainer import reporter as reporter_module


class MyEvaluator(extensions.Evaluator):
    """
    chainer標準のEvaliator(testデータの評価出力)で
    取るべきでないのに各バッチの平均が取られてしまう問題を修正したクラス
    """
    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None):
        super(MyEvaluator, self).__init__(iterator, target, converter, device, eval_hook, eval_func)

    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()
        freq_errs = dict()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                with F.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        eval_func(*in_arrays)
                    elif isinstance(in_arrays, dict):
                        eval_func(**in_arrays)
                    else:
                        eval_func(in_arrays)

            summary.add(observation)
            freq_errs[observation['validation/main/freq_err']] = \
                freq_errs.get(observation['validation/main/freq_err'], 0) + 1

        d = {name: summary.compute_mean() for name, summary in six.iteritems(summary._summaries)}
        d['validation/main/freq_err'] = max([(v, k) for k, v in freq_errs.items()])[1]
        return {'val/' + name.split('/')[-1]: sammary for name, sammary in d.items()}


class Block(chainer.Chain):
    """
    畳み込み層
    """
    def __init__(self, out_channels, ksize, stride=1, pad=0):
        super(Block, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, out_channels, ksize, stride, pad)
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
        with self.init_scope():
            # self.block1 = Block(32, 5, pad=1)  # n_in = args.size (300)^2 * 3 = 270000
            # self.block2 = Block(64, 3, pad=1)
            # self.block3 = Block(128, 3, pad=1)
            # self.block4 = Block(256, 3, pad=1)
            # self.block5 = Block(128, 3, pad=1)

            self.block1 = Block(32, 3)  # n_in = args.size (300)^2 * 3 = 270000
            self.block2 = Block(64, 2)
            self.block3 = Block(128, 2)
            self.block4 = Block(256, 2)
            self.block5 = Block(256, 2)
            self.block6 = Block(128, 2)

            self.fc1 = L.Linear(512)
            self.fc2 = L.Linear(512)
            # ↓中身を調べている最中
            # self.bn_fc1 = L.BatchNormalization(512)
            self.fc3 = L.Linear(n_out)

    def loss_func(self, x, t):
        y = self.predict(x)
        t_card = F.sum(t.astype("f"), axis=1)

        TP = F.sum((y+1) * 0.5 * t, axis=1)
        FP = F.sum((y+1) * 0.5 * (1 - t), axis=1)
        FN = F.sum((1 - y) * 0.5 * t, axis=1)

        loss = 1 - F.average(2 * TP / (2 * TP + FP + FN))  # F1 scoreを元に
        loss += F.average(F.sum(t * F.exp(- y), axis=1) * F.sum((1 - t) * F.exp(y), axis=1) /
                         (t_card * (t.shape[1] - t_card)))  # https://ieeexplore.ieee.org/document/1683770/ (3)式を変形

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
        # 64 channel blocks:
        # h = self.block1(x)
        # h = F.max_pooling_2d(h, 3)
        # h = self.block2(h)
        # h = F.max_pooling_2d(h, 3)
        # h = self.block3(h)
        # h = F.max_pooling_2d(h, 2)
        # h = self.block4(h)
        # h = F.max_pooling_2d(h, 2)
        # h = self.block5(h)

        h = self.block1(x)
        h = F.max_pooling_2d(h, 2)
        h = self.block2(h)
        h = F.max_pooling_2d(h, 2)
        h = self.block3(h)
        h = F.max_pooling_2d(h, 2)
        h = self.block4(h)
        h = F.max_pooling_2d(h, 2)
        h = self.block5(h)
        h = F.max_pooling_2d(h, 2)
        h = self.block6(h)


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
            ax.plot(range(1, self.n + 1), j)
            for o in [5, 20]:  # 平均移動線
                if self.n >= o:
                    average_line = [0] * (self.n - o + 1)
                    for k, l in enumerate(j):
                        for m in range(o):
                            if k - m >= 0 and k - m < len(average_line):
                                average_line[k - m] += l
                    ax.plot(range(o, self.n + 1), list(map(lambda y: y / o, average_line)))

            if i == 0:
                ax.set_ylim(None, 1.6)
            elif i == 2 or i == 3:
                ax.set_ylim(0.7, None)
            #ax.set_xticks(range(1, self.n + 1))
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


class Transform(object):
    def __init__(self, args):
        self.label_variety = args.label_variety
        self.size = args.size
        with open('input/train.json', 'r') as f:
            self.json_data = [[int(j) for j in i["labelId"]] for i in
                              json.load(f)["annotations"][:args.total_photo_num]]
        self.data_folder = 'data/' + args.object + '_images/'
        self.file_nums = os.listdir(self.data_folder)
        map(lambda x: self.file_nums.remove(x), ['.keep', '.DS_Store', 'trash'])
        self.file_nums = [int(i.split('.')[0]) for i in self.file_nums]
        self.file_nums.sort()

    def __call__(self, num):
        img_data = Image.open(self.data_folder + str(self.file_nums[num]) + '.jpg')
        img_data = img_data.resize([self.size] * 2, Image.ANTIALIAS)  # 画像を一定サイズに揃える
        array_img = np.asarray(img_data).transpose(2, 0, 1).astype(np.float32) / 255.  # データを整えて各値を0~1の間に収める
        array_img = self.augment(img_data, array_img)
        one_hot_label = np.array([1 if i in self.json_data[num] else 0 for i in range(1, self.label_variety + 1)])
        # すべてのlabel番号に対しlebelがついているならば1,そうでないならば0を入れたリスト
        # 例: 1, 2, 10 のラベルがついている場合
        # [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, ...]
        return array_img, one_hot_label

    def augment(self, raw, img):
        if np.random.rand() < 0.5:
            img = img[::-1, :, :]  # 左右を逆に
        if np.random.rand() < 0.2:
            angle = np.random.randint(-30, 30)
            img = rotate(img, angle, axes=(1, 2))
            # img = raw.rotate(angle)
            # img = np.asarray(img2).transpose(2, 0, 1).astype(np.float32) / 255.
            img = imresize(img.transpose(1, 2, 0), [self.size] * 2).transpose((2, 0, 1))

        # mpl.pyplot.imshow(img.transpose(1, 2, 0))
        return img


def main():
    #各種パラメータ設定
    parser = argparse.ArgumentParser(description='Linear iMaterialist_Challenge:')
    parser.add_argument('--batchsize', '-b', type=int, default=256,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',  # result/resume.npz',
                        help='Resume the training from snapshot')
    parser.add_argument('--early-stopping', type=str,
                        help='Metric to watch for early stopping')
    parser.add_argument('--frequency', '-f', type=int, default=1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension'),
    parser.add_argument('--size', type=int, default=128),  # 正規化する時の一辺のpx
    parser.add_argument('--label_variety', type=int, default=228),  # 確認できたlabelの総数 この中で判断する
    parser.add_argument('--total_photo_num', type=int, default=9815),  # 使用する写真データの数(9815, 39269)
    parser.add_argument('--object', type=str, default='train')  # train or test のどちらか選んだ方のデータを使用する
    args = parser.parse_args()

    # モデルの定義
    model = Mymodel(args.label_variety)

    # GPUで動かせるのならば動かす
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # optimizerのセットアップ
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # データセットのセットアップ
    dataset = TransformDataset(range(args.total_photo_num), Transform(args))
    train, test = chainer.datasets.split_dataset_random(dataset, int(args.total_photo_num * 0.8),
                                                        seed=0)  # 2割をvalidation用にとっておく

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # 学習をどこまで行うかの設定
    stop_trigger = (args.epoch, 'epoch')
    if args.early_stopping:  # iptimizerがAdamだと無意味
        stop_trigger = training.triggers.EarlyStoppingTrigger(
            monitor=args.early_stopping, verbose=True,
            max_trigger=(args.epoch, 'epoch'))

    # uodater, trainerのセットアップ
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu, loss_func=model.loss_func)
    trainer = training.Trainer(updater, stop_trigger, out=args.out)

    # testデータでの評価の設定
    evaluator = MyEvaluator(test_iter, model, device=args.gpu, eval_func=model.loss_func)
    evaluator.trigger = 1, 'epoch'
    trainer.extend(evaluator)

    # モデルの層をdotファイルとして出力する設定
    trainer.extend(extensions.dump_graph('main/loss'))

    # snapshot(学習中の重み情報)の保存
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # trainデータでの評価の表示頻度設定
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))

    # 各データでの評価の保存設定
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss', 'val/loss'],
                'iteration', trigger=(5, 'iteration'), file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/acc', 'val/acc'],
                'iteration', trigger=(5, 'iteration'), file_name='accuracy.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/freq_err', 'val/freq_err'],
                'iteration', trigger=(5, 'iteration'), file_name='frequent_error.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/acc2', 'val/acc2'],
                'iteration', trigger=(5, 'iteration'), file_name='accuracy2.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/f1', 'val/f1'],
                'iteration', trigger=(5, 'iteration'), file_name='f1.png'))

    # 各データでの評価の表示(欄に関する)設定
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'val/loss',
         'main/acc', 'vali/acc',
         'main/freq_err', 'val/freq_err', 'main/acc_66',
         'main/acc2', 'val/acc2', 'main/f1', 'val/f1', 'elapsed_time'
         ]))

    #プログレスバー表示の設定
    trainer.extend(extensions.ProgressBar())

    # 学習済みデータの読み込み設定
    if os.path.isfile(args.resume) and args.resume:
        pass
    # chainer.serializers.load_npz("result/snapshot_iter_63", trainer)
    # chainer.serializers.load_npz("result/snapshot_iter_0", model, path='updater/model:main/')

    # 学習の実行
    trainer.run()


if __name__ == '__main__':
    main()

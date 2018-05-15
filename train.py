#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings

warnings.filterwarnings('ignore', category=FutureWarning, message="Conversion of the second")
warnings.filterwarnings('ignore', category=RuntimeWarning, message="invalid value encountered in sqrt")

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.datasets import TransformDataset
import os
import numpy as np
from PIL import Image
import json
import argparse

from chainer import training
from chainer.training import extensions


class Block(chainer.Chain):
    """
    畳み込み層
    """
    def __init__(self, out_channels, ksize, stride=1, pad=1):
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
        super(Mymodel, self).__init__()
        with self.init_scope():
            self.block1_1 = Block(64, 8, 2, 2)  # n_in = args.size (300)^2 * 3 = 270000
            self.block1_2 = Block(64, 5)
            self.block2_1 = Block(128, 3)
            self.block2_2 = Block(128, 3)
            self.block3_1 = Block(256, 3)
            self.block3_2 = Block(256, 3)
            self.block4_1 = Block(512, 3)
            self.block4_2 = Block(256, 3)

            self.fc1 = L.Linear(4096)
            self.fc2 = L.Linear(2048)
            #↓中身を調べている最中
            #self.bn_fc1 = L.BatchNormalization(512)
            self.fc3 = L.Linear(n_out)

    def loss_func(self, x, t):
        y = self.predict(x)

        loss = np.sum(- np.log(np.absolute(y + t - 1))) / len(x)
        # labelが付いている(t_が1)場合:   -log(y_)
        #      付いていない(t_が0)場合:   -log(1-y_)     ここでt_,y_ はx, yの要素
        # 以上の総和をバッチサイズで割る

        chainer.reporter.report({'loss': loss}, self)
        accuracy = self.accuracy(y, t)
        chainer.reporter.report({'accuracy': accuracy[0]}, self)  # dataひとつひとつのlabelが完全一致している確率
        chainer.reporter.report({'frequent_error': accuracy[1]}, self)  # batchの中で最も多く間違って判断したlabel
        return loss

    def accuracy(self, y, t):
        y_binary = (y.data > 0.5).astype(int)
        accuracy1 = sum([1 if all(i) else 0 for i in (y_binary == t)]) / len(y)  # dataひとつひとつのlabelが完全一致している確率
        return accuracy1, np.sum((y_binary != t).astype(int), 0).argsort()[-1] + 1

    def predict(self, x):
        # 64 channel blocks:
        h = self.block1_1(x)
        h = F.dropout(h, ratio=0.3)
        h = self.block1_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 128 channel blocks:
        h = self.block2_1(h)
        h = F.dropout(h, ratio=0.3)
        h = self.block2_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 256 channel blocks:
        h = self.block3_1(h)
        h = F.dropout(h, ratio=0.3)
        h = self.block3_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block4_1(h)
        h = F.dropout(h, ratio=0.3)
        h = self.block4_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.dropout(h, ratio=0.4)
        h = self.fc1(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)
        h = self.fc2(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)  # dropout 多すぎる？
        return F.sigmoid(self.fc3(h))


class Transform(object):
    def __init__(self, args, json_data):
        self.label_variety = args.label_variety
        self.size = args.size
        self.json_data = json_data
        self.data_folder = 'data/' + args.object + '_images/'

    def __call__(self, num):
        img_data = Image.open(self.data_folder + str(num + 1) + '.jpg')
        img_data = img_data.resize([self.size] * 2, Image.ANTIALIAS)  # 画像を一定サイズに揃える
        array_img = np.asarray(img_data).transpose(2, 0, 1).astype(np.float32) / 255.  # データを整えて各値を0~1の間に収める

        label = [1 if i in self.json_data[num] else 0 for i in range(self.label_variety)]
        # すべてのlabel番号に対しlebelがついているならば1,そうでないならば0を入れたリスト
        #
        # 例: 1, 2, 10 のラベルがついている場合
        # [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, ...]

        return array_img, label


def main():
    parser = argparse.ArgumentParser(description='Linear iMaterialist_Challenge:')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',  # result/resume.npz',
                        help='Resume the training from snapshot')
    parser.add_argument('--early-stopping', type=str,
                        help='Metric to watch for early stopping')
    parser.add_argument('--frequency', '-f', type=int, default=20,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=256,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension'),
    parser.add_argument('--size', type=int, default=300),  # 正規化する時の一辺のpx
    parser.add_argument('--label_variety', type=int, default=228),  # 確認できたlabelの総数 この中で判断する
    parser.add_argument('--total_photo_num', type=int, default=20000),  # 使用する写真データの数
    parser.add_argument('--object', type=str, default='train')  # train or test のどちらか選んだ方のデータを使用する
    args = parser.parse_args()


    model = Mymodel(args.label_variety)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    with open('input/train.json', 'r') as f:
        json_data = np.array([[int(j) for j in i["labelId"]] for i in json.load(f)["annotations"][:args.total_photo_num]])

    dataset = TransformDataset(range(args.total_photo_num), Transform(args, json_data))
    train, test = chainer.datasets.split_dataset_random(dataset, int(args.total_photo_num * 0.8), seed=3110)
    # 2割をvalidation用にとっておく

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    stop_trigger = (args.epoch, 'epoch')
    # Early stopping option
    if args.early_stopping:
        stop_trigger = chainer.training.triggers.EarlyStoppingTrigger(
            monitor=args.early_stopping, verbose=True,
            max_trigger=(args.epoch, 'epoch'))

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu, loss_func=model.loss_func)
    trainer = training.Trainer(updater, stop_trigger, out=args.out)

    evaluator = extensions.Evaluator(test_iter, model, device=args.gpu, eval_func=model.loss_func)
    evaluator.trigger = 50, 'iteration'
    trainer.extend(evaluator)

    trainer.extend(extensions.dump_graph('main/loss'))

    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'iteration'))

    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))

    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', trigger=(1, 'iteration'), file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', trigger=(1, 'iteration'), file_name='accuracy.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy2', 'validation/main/accuracy2'],
                'epoch', trigger=(1, 'iteration'), file_name='accuracy2.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/frequent_error', 'validation/main/frequent_error'],
                'epoch', trigger=(1, 'iteration'), file_name='frequent_error.png'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'main/accuracy2', 'validation/main/accuracy2',
         'main/frequent_error', 'elapsed_time']))

    trainer.extend(extensions.ProgressBar())

    if os.path.isfile(args.resume) and args.resume:
        pass
        #chainer.serializers.load_npz("result/snapshot_iter_0", trainer)
        #chainer.serializers.load_npz("result/snapshot_iter_0", model, path='updater/model:main/')

    # Run the training
    trainer.run()

    # chainer.serializers.save_npz("resume.npz", model)#学習データの保存


if __name__ == '__main__':
    main()

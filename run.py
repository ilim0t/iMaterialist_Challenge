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
from chainer.dataset import convert
import tqdm

from chainer import training
from chainer.training import extensions


class Block(chainer.Chain):
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
            self.block1_1 = Block(64, 8, 2, 2)  # n_in = args.size (300) ^ 2 * 3 = 270000 から
            self.block1_2 = Block(64, 5)
            self.block2_1 = Block(128, 3)
            self.block2_2 = Block(128, 3)
            self.block3_1 = Block(256, 3)
            self.block3_2 = Block(256, 3)
            self.block4_1 = Block(512, 3)
            self.block4_2 = Block(256, 3)

            self.fc1 = L.Linear(4096)
            self.fc2 = L.Linear(2048)
            # self.bn_fc1 = L.BatchNormalization(512)
            self.fc3 = L.Linear(n_out)

    def __call__(self, x, *args):
        y = self.predict(x)
        return y.data > 0.5

    def loss_func(self, x, t):
        y = self.predict(x)
        loss = F.sum((y-t) * (y-t)) / len(x)
        chainer.reporter.report({'loss': loss}, self)
        accuracy = self.myaccuracy(y, t)
        chainer.reporter.report({'accuracy': accuracy[0]}, self)
        chainer.reporter.report({'accuracy2': accuracy[1]}, self)
        chainer.reporter.report({'frequent_error': accuracy[2]}, self)
        return loss

    def myaccuracy(self, y, t):
        y_binary = (y.data > 0.5).astype(int)
        #accuracy1はFalse Positiveが多すぎる
        accuracy1 = sum([1 if all(i) else 0 for i in (y_binary == t)]) / len(y)  # batchのコードが完全一致している確率
        accuracy2 = sum(sum((y_binary == t).astype(int))) / len(y) / len(y[0])  # すべてのbatchを通してlabelそれぞれの正解確率の平均
        return accuracy1, accuracy2, np.sum((y_binary != t).astype(int), 0).argsort()[-1] + 1

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
        h = F.dropout(h, ratio=0.5)
        return F.sigmoid(self.fc3(h))


class Transform(object):
    data_folder = 'data/train_images/'

    def __init__(self, args):
        self.label_variety = args.label_variety
        self.size = args.size
        self.data_folder = 'data/' + args.object + '_images/'

    def __call__(self, num):
        print(num)
        img_data = Image.open(self.data_folder + str(num + 1) + '.jpg')
        img_data = img_data.resize([self.size] * 2, Image.ANTIALIAS)
        array_img = np.asarray(img_data).transpose(2, 0, 1).astype(np.float32) / 255.
        return array_img


def main():
    parser = argparse.ArgumentParser(description='Linear iMaterialist_Challenge:')
    parser.add_argument('--batchsize', '-b', type=int, default=40,
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
    parser.add_argument('--size', type=int, default=300),
    parser.add_argument('--label_variety', type=int, default=228),
    parser.add_argument('--total_photo_num', type=int, default=39706),
    parser.add_argument('--object', type=str, default='train'),
    args = parser.parse_args()


    model = Mymodel(args.label_variety)

    dataset = TransformDataset(range(args.total_photo_num), Transform(args))
    test = chainer.datasets.SubDataset(dataset, 1, args.total_photo_num)
    test_iter = chainer.iterators.SerialIterator(test, 10, shuffle=False)

    chainer.serializers.load_npz('result/snapshot_iter_200', model, path='updater/model:main/')


    pbar = tqdm.tqdm(total=args.total_photo_num)

    with chainer.using_config('train', False):
        batch = test_iter.next()
        in_arrays = convert.concat_examples(batch, -1)
        y = model(in_arrays)
        pbar.update(1)
        import pprint
        pprint.pprint(y)
        print(y)

    pbar.close()

if __name__ == '__main__':
    main()

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
from chainer import training
from chainer.training import extensions
from chainer.dataset import convert

import train
import mymodel

import numpy as np
import argparse
import pickle
import os
import json
from PIL import Image
import matplotlib as mpl
import platform

# for rendering graph on remote server.
if platform.system() != "Darwin":
    mpl.use('Agg')
import matplotlib.pyplot as plt

import tqdm
import pandas as pd


def main():
    # 各種パラメータ設定
    parser = argparse.ArgumentParser(description='iMaterialist_Challenge:')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='1バッチあたり何枚か')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='何epochやるか')
    parser.add_argument('--out', '-o', default='result',
                        help='結果を出力するディレクトリ')
    parser.add_argument('--resume', '-r', default='',
                        help='指定したsnapshopから継続して学習します')
    parser.add_argument('--frequency', '-f', type=int, default=1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='使うGPUの番号')
    parser.add_argument('--size', '-s', type=int, default=128,
                        help='正規化する時の一辺のpx'),
    parser.add_argument('--label_variety', type=int, default=228,
                        help='確認できたlabelの総数 この中で判断する'),
    parser.add_argument('--total_photo_num', '-n', type=int, default=-1,
                        help='使用する写真データの数'),  # (9815, 39269)
    parser.add_argument('--object', type=str, default='test',
                        help='train or test のどちらか選んだ方のデータを使用する'),
    parser.add_argument('--cleanup', '-c', dest='cleanup', action='store_true',
                        help='邪魔な画像を取り除く'),
    parser.add_argument('--interval', '-i', type=int, default=10,
                        help='何iteraionごとに画面に出力するか')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    model = mymodel.ResNet(args.label_variety)
    #model = mymodel.Mymodel(args.label_variety)

    # GPUで動かせるのならば動かす
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    photo_nums = train.photos(args)
    test = chainer.datasets.TransformDataset(photo_nums, train.Transform(args, photo_nums, False))
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    chainer.serializers.load_npz(args.resume, model, path='updater/model:main/')

    pbar = tqdm.tqdm(total=len(photo_nums))

    with chainer.using_config('train', False):
        list0 = []
        num = 1
        for batch in test_iter:
            in_arrays = convert.concat_examples(batch, args.gpu)
            y = model(in_arrays)
            for i in y.data > 0:
                list0.append([str(num), ' '.join((np.where(i)[0] + 1).astype(str))])
                num += 1
                pbar.update(1)
    pbar.close()

    df = pd.DataFrame(list0, columns=['image_id', 'label_id'])
    df.to_csv("result.csv", index=False)


if __name__ == '__main__':
    main()

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
                        help='指定したepotchごとに重みを保存します')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='使うGPUの番号')
    parser.add_argument('--size', '-s', type=int, default=256,
                        help='正規化する時の一辺のpx'),
    parser.add_argument('--label_variety', type=int, default=228,
                        help='確認できたlabelの総数 この中で判断する'),
    parser.add_argument('--total_photo_num', '-n', type=int, default=-1,
                        help='使用する写真データの数'),  # (9815, 39269)
    parser.add_argument('--object', type=str, default='test',
                        help='train or test のどちらか選んだ方のデータを使用する'),
    parser.add_argument('--cleanup', '-c', dest='cleanup', action='store_false',
                        help='付与すると 邪魔な画像を取り除き trashディレクトリに移動させる機能を停止させます'),
    parser.add_argument('--interval', '-i', type=int, default=10,
                        help='何iteraionごとに画面に出力するか')
    parser.add_argument('--model', '-m', type=int, default=0,
                        help='使うモデルの種類')
    parser.add_argument('--lossfunc', '-l', type=int, default=0,
                        help='使うlossの種類'),
    parser.add_argument('--stream', '-d', dest='stream', action='store_true',
                        help='画像のダウンロードを同時に行う'),
    parser.add_argument('--parallel', '-p', dest='douji', action='store_true',
                        help='画像ダウンロードを並列処理するか')
    args = parser.parse_args()

    # liteがついているのはsizeをデフォルトの半分にするの前提で作っています
    # RES_SPP_netはchainerで可変量サイズの入力を実装するのが難しかったので頓挫
    model = ['ResNet', 'ResNet_lite', 'Bottle_neck_RES_net', 'Bottle_neck_RES_net_lite',
             'Mymodel', 'RES_SPP_net', 'Lite'][args.model]

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# model: {}'.format(model))
    print('# size: {}'.format(args.size))
    print('')

    model = getattr(mymodel, model)(args.label_variety, args.lossfunc)

    # GPUで動かせるのならば動かす
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    photo_nums = train.photos(args)
    test = chainer.datasets.TransformDataset(
        photo_nums, train.Transform(args, photo_nums, False, False if args.model == 5 else True))
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    chainer.serializers.load_npz(args.resume, model, path='updater/model:main/')

    pbar = tqdm.tqdm(total=len(photo_nums))
    with chainer.using_config('train', False):
        list0 = []
        num = 1
        for batch in test_iter:
            try:
                in_arrays = convert.concat_examples(batch, args.gpu)
                y = model(in_arrays)
                for i in y.data > 0:
                    list0.append([str(num), ' '.join((np.where(i)[0] + 1).astype(str))])
                    num += 1
                    pbar.update(1)
            except Exception as e:
                print('')
                print(num)
                print(e)
                #print(batch)
                print('')
        pbar.close()

        df = pd.DataFrame(list0, columns=['image_id', 'label_id'])
        df.to_csv("result/result.csv", index=False)


if __name__ == '__main__':
    main()

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

from scipy.ndimage.interpolation import rotate
from scipy.misc import imresize

import shutil
import copy
import six


class MyEvaluator(extensions.Evaluator):
    """
    chainer標準のEvaliator(testデータの評価出力)で
    取るべきでないのに各バッチの平均が取られてしまう問題を修正し
    省略したPrintReportに対応したクラス
    """
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

        summary = chainer.reporter.DictSummary()
        freq_errs = dict()

        for batch in it:
            observation = {}
            with chainer.reporter.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                with chainer.function.no_backprop_mode():
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


class Transform(object):
    def __init__(self, args, photo_nums, isTrain=True):
        self.label_variety = args.label_variety
        self.size = args.size
        self.data_folder = 'data/' + args.object + '_images/'

        with open('input/train.json', 'r') as f:  # 教師データの読み込み
            self.json_data = [[int(j) for j in i["labelId"]] for i in
                              json.load(f)["annotations"][:photo_nums[-1 if args.total_photo_num == -1
                              else args.total_photo_num - 1]]]

    def __call__(self, num):
        img_data = Image.open(self.data_folder + str(num) + '.jpg')
        array_img = np.asarray(img_data)

        # 各種 augmentation
        if np.random.rand() < 0.5:
            array_img = self.rotation(array_img)
        if (array_img[:2] != [self.size]).any():
            array_img = imresize(array_img, [self.size] * 2)
        array_img = self.horizontal_flip(array_img)

        array_img = self.zscore(array_img)
        array_img = array_img.transpose(2, 0, 1).astype(np.float32)

        label = np.array([1 if i in self.json_data[num - 1] else 0 for i in range(1, self.label_variety + 1)])
        # すべてのlabel番号に対しlebelがついているならば1,そうでないならば0を入れたリスト
        # 例: 1, 2, 10 のラベルがついている場合
        # [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, ...]
        return array_img, label

    def horizontal_flip(self, img):
        # 左右反転
        return img[:, ::-1, :]

    def rotation(self, img):
        # 回転
        angle = np.random.randint(-20, 20)
        img = rotate(img, angle)
        return img

    def zscore(self, img):
        # 正規化
        xmean = np.mean(img)  # 110~170
        xstd = np.std(img)  # 80~110
        img = (img - xmean) / xstd
        return img


def photos(args):
    # 存在するファイルから写真のみ列挙してself.file_numsに格納
    data_folder = 'data/' + args.object + '_images/'
    photo_nums = os.listdir(data_folder)
    for i in ['.gitkeep', '.DS_Store', 'trash']:  # 画像ではないファイル,ディレクトリを除外
        if i in photo_nums:
            photo_nums.remove(i)
    photo_nums = [int(i.split('.')[0]) for i in photo_nums]
    photo_nums.sort()

    if args.cleanup and args.object != 'test':  # 指定された場合 真っ白なファイルなどを除外する
        removed = []
        for i in photo_nums:
            # [4019, 16161, 35, 1485, 7742, 34262, 5098] 40000枚中これらのみ発見
            if os.path.getsize(data_folder + str(i) + '.jpg') in [874, 1854, 2588, 5106, 11197, 12814, 6305]:
                photo_nums.remove(i)
                removed.append(i)
                shutil.move(data_folder + str(i) + '.jpg', data_folder + 'trash/' + str(i) + '.jpg')
        with open('removed.txt', 'wb') as f:
            pickle.dump(removed, f)

    if args.total_photo_num != -1:  # 上限が指定されている場合それに合わせる
        photo_nums = photo_nums[:args.total_photo_num]
    return photo_nums

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
    parser.add_argument('--object', type=str, default='train',
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

    # モデルの定義
    model = mymodel.ResNet(args.label_variety)
    #model = models.model.Mymodel(args.label_variety)

    # GPUで動かせるのならば動かす
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # optimizerのセットアップ
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # データセットのセットアップ
    photo_nums = photos(args)
    train, test = chainer.datasets.split_dataset_random(photo_nums,
                                                        int(len(photo_nums) * 0.8), seed=0)  # 2割をvalidation用にとっておく
    train = chainer.datasets.TransformDataset(train, Transform(args, photo_nums, True))
    test = chainer.datasets.TransformDataset(test, Transform(args, photo_nums, False))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # 学習をどこまで行うかの設定
    stop_trigger = (args.epoch, 'epoch')
    # if args.early_stopping:  # optimizerがAdamだと無意味
    #     stop_trigger = training.triggers.EarlyStoppingTrigger(
    #         monitor=args.early_stopping, verbose=True,
    #         max_trigger=(args.epoch, 'epoch'))

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
    trainer.extend(extensions.LogReport(trigger=(args.interval, 'iteration')))

    # 各データでの評価の保存設定
    if extensions.PlotReport.available():
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
         'main/acc', 'val/acc', 'main/f1', 'val/f1', 'elapsed_time']))
    # trainer.extend(extensions.PrintReport(
    #     ['epoch', 'iteration', 'main/loss', 'val/loss',
    #      'main/acc', 'val/acc', 'main/acc2', 'val/acc2',
    #      'main/acc_66', 'main/f1', 'val/f1',
    #      'main/freq_err', 'val/freq_err', 'elapsed_time'
    #      ]))


    # プログレスバー表示の設定
    trainer.extend(extensions.ProgressBar(update_interval=args.interval))

    # 学習済みデータの読み込み設定
    if args.resume:
        chainer.serializers.load_npz(args.resume, model)

    # 学習の実行
    trainer.run()


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-


import warnings

warnings.filterwarnings('ignore', category=FutureWarning, message="Conversion of the second")
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
import time

# for rendering graph on remote server.
if platform.system() != "Darwin":
    mpl.use('Agg')
import matplotlib.pyplot as plt

from scipy.ndimage import rotate
from scipy.misc import imresize

import shutil
import copy
import six

import io
import urllib3

warnings.filterwarnings('ignore', category=urllib3.exceptions.InsecureRequestWarning,
                        message="Unverified HTTPS request is being made.")

# サーバで動かすときに必要なパッケージ
# pillow
# scipy


class MyShift(chainer.training.extensions.LinearShift):
    def __init__(self, attr, rate, logreport, init=None, optimizer=None):
        self._attr = attr
        if rate < 0:
            raise ValueError('ExponentialShift does not support negative rate')
        self._rate = rate
        self.logreport = logreport
        self._init = init
        self._optimizer = optimizer
        self._t = 0
        self.n = -20
        self._last_value = None

    def __call__(self, trainer):
        self._t += 1
        self.n += 1
        optimizer = self._get_optimizer(trainer)
        value = self._compute_next_value(optimizer)
        self._update_value(optimizer, value)

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)
        # ensure that _init is set
        if self._init is None:
            self._init = getattr(optimizer, self._attr)

        if self._last_value is not None:  # resuming from a snapshot
            self._update_value(optimizer, self._last_value)
        else:
            self._update_value(optimizer, self._init)

    def _compute_next_value(self, optimizer):
        time.sleep(1)
        if self.n < 20:
            return self._last_value
        if np.corrcoef([range(20), [i['main/loss'] for i in self.logreport.log[-20:]]])[0][1] > -0.2:  # 相関係数
            print("!!!!レート下げたよ!!!!相関係数:", np.corrcoef([range(10), [i['main/loss'] for i in self.logreport.log[-20:]]])[0][1])
            self.n = 0
            return self._last_value * self._rate
        return self._last_value


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
            if 'validation/main/freq_err' in observation.keys():
                freq_errs[observation['validation/main/freq_err']] = \
                    freq_errs.get(observation['validation/main/freq_err'], 0) + 1

        d = {name: summary.compute_mean() for name, summary in six.iteritems(summary._summaries)}
        if 'validation/main/freq_err' in observation.keys():
            d['validation/main/freq_err'] = max([(v, k) for k, v in freq_errs.items()])[1]
        return {'val/' + name.split('/')[-1]: sammary for name, sammary in d.items()}


class Transform(object):
    def __init__(self, args, photo_nums, isTrain=True, isResize=True):
        self.label_variety = args.label_variety
        self.size = args.size
        self.data_folder = 'data/' + args.object + '_images/'
        self.isTrain = isTrain
        self.isResize = isResize
        self.stream = args.stream
        self.cleanup = args.cleanup
        self.model = args.model

        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

        if self.stream:
            with open('input/train.json', 'r') as f:  # 教師データの読み込み
                data = json.load(f)
                self.url = [i["url"] for i in data["images"]]
                self.json_data = [[int(j) for j in i["labelId"]] for i in data["annotations"]]
            self.randnums = list(range(1014544 + 1))
            np.random.shuffle(self.randnums)
            self.cast = {}
        elif isTrain:
            with open('input/train.json', 'r') as f:  # 教師データの読み込み
                self.json_data = [[int(j) for j in i["labelId"]] for i in
                                  json.load(f)["annotations"][:photo_nums[-1 if args.total_photo_num == -1 else
                                  args.total_photo_num - 1]]]

    def __call__(self, num):
        if num < 0 and self.stream:
            num = - num
            img_data = self.download(num)
        # 写真を読み込み配列に
        else:
            img_data = Image.open(self.data_folder + str(num) + '.jpg')
        array_img = np.asarray(img_data)

        # 各種 augmentation
        array_img = self.crop(array_img)
        # if (array_img[:2] != [self.size]).any():
        #     array_img = imresize(array_img, [self.size] * 2)
        if np.random.rand() < 0.5:
            array_img = self.rotation(array_img)
        array_img = self.horizontal_flip(array_img)

        array_img = self.zscore(array_img)
        # mpl.pyplot.imshow(array_img)  # 表示
        if self.model == 6 or self.model == 7:
            array_img = L.model.vision.vgg.prepare(array_img)
        else:
            array_img = array_img.transpose(2, 0, 1).astype(np.float32)
        if not self.isTrain:
            return array_img

        label = np.array([1 if i in self.json_data[num - 1] else 0 for i in range(1, self.label_variety + 1)])
        # すべてのlabel番号に対しlebelがついているならば1,そうでないならば0を入れたリスト
        # 例: 1, 2, 10 のラベルがついている場合
        # [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, ...]
        return array_img, label

    def download(self, num):
        if num in self.cast.keys():
            return Image.open(self.data_folder + str(self.cast[num]) + '.jpg')
        while 1:
            while 1:  # 並列化のため複雑化
                casetdnum = self.randnums[0]
                if casetdnum != -1:
                    self.randnums[0] = -1
                    break
            img = photo_downloader(self.data_folder + str(casetdnum + 1) + '.jpg', self.url[casetdnum])
            trash = not self.cleanup or cleanup(self.data_folder, casetdnum + 1)
            del self.randnums[0]
            if not trash:
                self.cast[num] = casetdnum + 1
                return img

    def horizontal_flip(self, img):  # 左右反転
        return img[:, ::-1, :]

    def rotation(self, img):  # 回転, 処理が重い(RES_netにて処理時間全体の3%)
        angle = np.random.randint(-10, 10)
        img = rotate(img, angle, cval=255)
        if self.isResize:
            img = imresize(img, [self.size] * 2)
        return img

    def zscore(self, img):  # 正規化(準備中)
        # xmean = np.mean(img)  # 110~170
        # xstd = np.std(img)  # 80~110
        # img = (img - xmean) / xstd
        return img / 255.

    def crop(self, img):  # 切り抜き, サイズ合わせ
        if not self.isResize:
            if img.shape[0] > self.size * 2 and img.shape[1] > self.size * 2:
                img = imresize(img, (img.shape[0] // 2, img.shape[1] // 2))
            return img
        ratio = max(img.shape[:2]) / min(img.shape[:2])
        if ratio >= 4 or img.shape[0] <= self.size or img.shape[1] <= self.size:
            # 少なくとも片辺の長さがが規定以下のとき
            # アス比が4:1以上ならばちぎって横に並べて(文字の折返しみたいに)長方形にした後長辺が最大になるようにトリミングしあまりを白で埋める
            # アス比がそれ以下ならそのまま長辺が最大になるように拡大しあまりを白で埋める
            if ratio >= 4:
                img = self.divide(img, int(ratio / 2))
            img = self.assign(img)
        else:
            # 両辺ともに既定値以上ならば短辺×0.8以上の枠でランダムに切り出す
            size = self.size if max(self.size, int(min(img.shape[:2])) * 0.8) == min(img.shape[:2]) else\
                np.random.randint(max(self.size, int(min(img.shape[:2])) * 0.8), min(img.shape[:2]))
            top = 0 if 0 == img.shape[0] - size else np.random.randint(0, img.shape[0] - size)
            left = 0 if 0 == img.shape[1] - size else np.random.randint(0, img.shape[1] - size)
            img = img[top:top + size, left:left + size, :]
            img = imresize(img, [self.size] * 2)
        return img

    def divide(self, img, n):  # ちぎって横に並べて(文字の折返しみたいに)長方形にする
        # n: n回分割する

        # サイズが分割数で割り切れないとエラーが出るため揃える
        size = int(np.max(img.shape[:2]) / n) * n
        start = int((np.max(img.shape[:2]) - size) / 2)
        if np.argmax(img.shape[:2]) == 0:
            img = img[start:start + size, :, :]
        else:
            img = img[:, start:start + size, :]
        # 分割してずらし結合
        imgs = np.split(img, n, axis=np.argmax(img.shape[:2]))
        img = np.concatenate(imgs, axis=np.argmin(img.shape[:2]))
        return img

    def assign(self, img):  # 長辺が最大になるように拡大しあまりを白で埋める

        # 長辺を最大化するようにリサイズ
        if np.argmax(img.shape[:2]) == 0:
            size = (self.size, int(img.shape[1] * self.size / img.shape[0]))
        else:
            size = (int(img.shape[0] * self.size / img.shape[1]), self.size)
        img = imresize(img, size)

        # 短辺の両端を白く穴埋めして規定値に揃える
        if img.shape[0] != self.size:
            start = int((self.size - img.shape[0]) / 2)
            imgs = [np.full((start, img.shape[1], 3), 255), img,
                    np.full((self.size - img.shape[0] - start, img.shape[1], 3), 255)]
            img = np.concatenate(imgs, axis=0)
        if img.shape[1] != self.size:
            start = int((self.size - img.shape[1]) / 2)
            imgs = [np.full((self.size, start, 3), 255), img,
                    np.full((self.size, self.size - img.shape[1] - start, 3), 255)]
            img = np.concatenate(imgs, axis=1)
        return img


def photo_downloader(fname, url):
    if not os.path.exists(fname):
        http = urllib3.PoolManager(retries=urllib3.util.Retry(connect=3, read=2, redirect=3))
        response = http.request("GET", url)
        image = Image.open(io.BytesIO(response.data))
        image_rgb = image.convert("RGB")
        image_rgb.save(fname, format='JPEG', quality=90)
        return image
    else:
        return Image.open(fname)


def photos(args):
    # 存在するファイルから写真のみ列挙して返す
    data_folder = 'data/' + args.object + '_images/'
    if args.stream:
        return range(-1, (-int(1014544 * 0.9) - 1) if args.total_photo_num == -1 else (-args.total_photo_num - 1), -1)
    photo_nums = os.listdir(data_folder)
    photo_nums = [int(i.split('.')[0]) for i in photo_nums
                  if '.' in i and i.split('.')[1] == 'jpg' and i.split('.')[0][0] != '.']
    photo_nums.sort()

    if args.cleanup and args.object != 'test':  # 指定された場合 真っ白なファイルなどを除外する
        removed = []
        for i in photo_nums:
            if cleanup(data_folder, i):
                photo_nums.remove(i)
                removed.append(i)
        # 見つかった除外対象写真一覧を保存
        with open('removed.txt', 'wb') as f:
            pickle.dump(removed, f)

    if args.total_photo_num != -1:  # 上限が指定されている場合それに合わせる
        photo_nums = photo_nums[:args.total_photo_num]
    return photo_nums


def cleanup(data_folder, photo_num):
    # 過去に見つかった除外対象写真と全く同じサイズのとき除外
    # その代表photo_num: [4019, 16161, 35, 1485, 7742, 34262, 5098] 40000枚中これらのみ発見
    if os.path.getsize(data_folder + str(photo_num) + '.jpg') in [874, 1854, 2588, 5106, 11197, 12814, 6305]:
        if not os.path.isdir(data_folder + 'trash/'):
            os.mkdir(data_folder + 'trash/')
        shutil.move(data_folder + str(photo_num) + '.jpg', data_folder + 'trash/' + str(photo_num) + '.jpg')
        return True
    return False


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
    parser.add_argument('--object', type=str, default='train',
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

    # args.model = -1
    # args.batchsize = 8
    args.size = 224
    # args.interval = 1
    # args.cleanup = False
    # args.lossfunc = 3
    # # args.stream = True
    # # args.total_photo_num = 200
    # args.resume = 'serverresult/snapshot_iter_74'

    # liteがついているのはsizeをデフォルトの半分にするの前提で作っています
    # RES_SPP_netはchainerで可変量サイズの入力を実装するのが難しかったので頓挫
    model = {
        0: 'ResNet',
        1: 'ResNet_lite',
        2: 'Bottle_neck_RES_net',
        3: 'Bottle_neck_RES_net_lite',
        4: 'Mymodel',
        5: 'RES_SPP_net',
        6: 'VGGTrans',
        7: 'RESNetTrans',
        8: 'Lite'
    }[args.model]

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# model: {}'.format(model))
    print('# size: {}'.format(args.size))
    print('')

    # モデルの定義
    model = getattr(mymodel, model)(args.label_variety, args.lossfunc)

    # GPUで動かせるのならば動かす
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # optimizerのセットアップ
    optimizer = chainer.optimizers.Adam()
    # optimizer = chainer.optimizers.MomentumSGD(0.1, 0.9)  # https://arxiv.org/pdf/1605.07146.pdf
    # chainer.optimizer.WeightDecay(0.0005)

    optimizer.setup(model)

    # データセットのセットアップ
    photo_nums = photos(args)
    train, val = chainer.datasets.split_dataset_random(
        photo_nums, int(len(photo_nums) * 0.8), seed=0)  # 2割をvalidation用にとっておく
    trans = Transform(args, photo_nums, True, False if args.model == 5 else True)
    train = chainer.datasets.TransformDataset(train, trans)
    val = chainer.datasets.TransformDataset(val, trans)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val_iter = chainer.iterators.SerialIterator(val, args.batchsize,
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
    evaluator = MyEvaluator(val_iter, model, device=args.gpu, eval_func=model.loss_func)
    evaluator.trigger = 1, 'epoch'
    trainer.extend(evaluator)

    if args.model == 6 or args.model == 7:
        model.base.disable_update()

    # モデルの層をdotファイルとして出力する設定
    trainer.extend(extensions.dump_graph('main/loss'))

    # snapshot(学習中の重み情報)の保存
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # trainデータでの評価の表示頻度設定
    logreport = extensions.LogReport(trigger=(args.interval, 'iteration'))
    trainer.extend(logreport)

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
        ['epoch', 'iteration', 'main/loss', 'val/loss', 'main/acc', 'main/acc2', 'val/acc2',
         'main/precision', 'main/recall', 'main/f1', 'val/f1', 'main/labelnum', 'main/fpk', 'elapsed_time']))
        # ['epoch', 'iteration', 'main/loss', 'val/loss','main/acc', 'val/acc', 'main/acc2', 'val/acc2',
        #  'main/precision', 'main/recall', 'main/f1', 'val/f1', 'main/labelnum', 'main/tpk', 'elapsed_time']))

    # プログレスバー表示の設定
    trainer.extend(extensions.ProgressBar(update_interval=args.interval))

    # trainer.extend(MyShift("lr", 1 / 5, logreport, 0.1))

    # 学習済みデータの読み込み設定
    if args.resume:
        chainer.serializers.load_npz(args.resume, model, path='updater/model:main/')  # なぜかpathを外すと読み込めなくなってしまった 原因不明

    # 学習の実行
    if args.stream and args.parallel:
        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        executor.submit(trainer.run)

        def train_download():
            for num in [train_iter.dataset._dataset[i] for i in train_iter._order]:
                trans.download(-num)
                time.sleep(0.01)

        def val_download():
            for num in train_iter.dataset._dataset[train_iter.dataset._start:]:
                trans.download(-num)
                time.sleep(0.01)
        executor.submit(train_download)
        executor.submit(val_download)
    else:
        trainer.run()


if __name__ == '__main__':
    main()

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

    def loss_func(self, x, t):
        y = self.predict(x)
        loss = F.sum((y-t) * (y-t)) / len(x)
        chainer.reporter.report({'loss': loss}, self)
        accuracy = self.myaccuracy(y, t)
        chainer.reporter.report({'accuracy': accuracy[0]}, self)
        chainer.reporter.report({'accuracy2': accuracy[1]}, self)
        return loss

    def myaccuracy(self, y, t):
        y_binary = (y.data > 0.5).astype(int)
        #accuracy1はFalse Positiveが多すぎる
        accuracy1 = sum([1 if all(i) else 0 for i in (y_binary == t)]) / len(y)  # batchのコードが完全一致している確率
        accuracy2 = sum(sum((y_binary == t).astype(int))) / len(y) / len(y[0])  # すべてのbatchを通してlabelそれぞれの正解確率の平均
        return accuracy1, accuracy2

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

    def __init__(self, args, json_data):
        self.label_variety = args.label_variety
        self.size = args.size
        self.json_data = [[int(j) for j in i["labelId"]] for i in json_data["annotations"][:args.total_photo_num]]

    def __call__(self, num):
        img_data = Image.open(self.data_folder + str(num + 1) + '.jpeg')
        img_data = img_data.resize([self.size] * 2, Image.ANTIALIAS)
        array_img = np.asarray(img_data).transpose(2, 0, 1).astype(np.float32) / 255.
        label = [1 if i in self.json_data[num] else 0 for i in range(self.label_variety)]
        return array_img, label


def main():
    parser = argparse.ArgumentParser(description='Linear iMaterialist_Challenge:')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',  # result/resume.npz',
                        help='Resume the training from snapshot')
    parser.add_argument('--early-stopping', type=str,
                        help='Metric to watch for early stopping')
    parser.add_argument('--frequency', '-f', type=int, default=5,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=256,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension'),
    parser.add_argument('--size', type=int, default=300),
    parser.add_argument('--label_variety', type=int, default=228),
    parser.add_argument('--total_photo_num', type=int, default=10000)
    args = parser.parse_args()


    model = Mymodel(args.label_variety)

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the dataset
    with open('input/train.json', 'r') as f:
        json_data = json.load(f)

    dataset = TransformDataset(range(args.total_photo_num), Transform(args, json_data))
    train, test = chainer.datasets.split_dataset_random(dataset, int(args.total_photo_num * 0.8), seed=0)  # 2割を検証用に

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    stop_trigger = (args.epoch, 'epoch')
    # Early stopping option
    if args.early_stopping:
        stop_trigger = chainer.training.triggers.EarlyStoppingTrigger(
            monitor=args.early_stopping, verbose=True,
            max_trigger=(args.epoch, 'iteration'))

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu, loss_func=model.loss_func)
    trainer = training.Trainer(updater, stop_trigger, out=args.out)

    # Evaluate the model with the test dataset for each epoch
    evaluator = extensions.Evaluator(test_iter, model, device=args.gpu, eval_func=model.loss_func)
    evaluator.trigger = 3, 'iteration'
    trainer.extend(evaluator)

    # Reduce the learning rate by half every 25 epochs.
    # trainer.extend(extensions.ExponentialShift('lr', 0.5),
    #                trigger=(25, 'epoch'))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'iteraion'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['loss', 'validation/loss'],
                                  'epoch', trigger=(1, 'epoch'), file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['accuracy', 'validation/accuracy'],
                'epoch', trigger=(1, 'epoch'), file_name='accuracy.png'))
        trainer.extend(
            extensions.PlotReport(
                ['accuracy2', 'validation/accuracy2'],
                'epoch', trigger=(1, 'epoch'), file_name='accuracy2.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'main/accuracy2', 'validation/main/accuracy2', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if os.path.isfile(args.resume) and args.resume:
        pass
        # chainer.serializers.load_npz("result/snapshot_iter_", model, path='updater/model:main/')

    # batch = test_iter.next()
    # from chainer.dataset import convert
    # in_arrays = convert.concat_examples(batch, -1)

    # Run the training
    trainer.run()

    # chainer.serializers.save_npz("resume.npz", model)#学習データの保存


if __name__ == '__main__':
    main()

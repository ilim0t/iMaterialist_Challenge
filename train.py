#!/usr/bin/env python


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

class Mymodel(chainer.Chain):
    data_folder = 'data/train_images/'

    def __init__(self, n_units, n_out):
        super(Mymodel, self).__init__()
        with self.init_scope():
            pass
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out
        
    def __call__(self, x, t):
        y = self.predict(x)
        loss = F.sum((y-t) * (y-t)) / len(x)
        chainer.reporter.report({'loss': loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return loss
        
    def predict(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return F.sigmoid(self.l3(h2))
            
def transform(num):
    img = Image.open('data/train_images/' + str(num + 1) + '.jpeg')
    img = img.resize((600,600), Image.ANTIALIAS)
    arrayImg = np.asarray(img).transpose(2,0,1).astype(np.float32)/255.    
    label = [int(i) for i in jsonData["annotations"][num]["labelId"]]
    label = [1 if i in label else 0 for i in  range(100)]#np型にする？
    return arrayImg, label

def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.2,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=3,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='resume.npz',
                        help='Resume the training from snapshot')
    parser.add_argument('--early-stopping', type=str,
                        help='Metric to watch for early stopping')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()


    dataset1 = TransformDataset(range(1000), transform)

    model = Mymodel(100, 100)


    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    chainer.links.Classifier

    # Load the MNIST dataset
    #train = lambda x: dataset(x)[0]
    #test =  lambda x: dataset(x)[1]

    train, test = chainer.datasets.split_dataset_random(dataset1, int(1000 * 0.8), seed=0)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)#, loss_func=F.softmax_cross_entropy)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)


    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    #trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if 0 and args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, model)
        print("resume on")


    # Run the training
    trainer.run()

    print("save resume")
    chainer.serializers.save_npz("resume.npz", model)#学習データの保存

global jsonData

if __name__ == '__main__':
    with open('input/train.json', 'r') as f:
        jsonData = json.load(f)
    main()
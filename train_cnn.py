#!/usr/bin/env python
"""
Cifar-10 classifier training using Convolution Neural Network.

"""

from __future__ import print_function
import argparse

import numpy as np
import six

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers

import time
import datahandler as dh


parser = argparse.ArgumentParser(description='Example: cifar-10')
parser.add_argument('--augmentation', '-a', default=1.5, type=float,
                    help='The amount of data augmentation')
parser.add_argument('--batchsize', '-b', default=100, type=int,
                    help='Batch size of training')
parser.add_argument('--data', '-d', choices=('on', 'off'),
                    default='off', help='Data normalization and augmentation flag')
parser.add_argument('--epoch', '-e', default=30, type=int,
                    help='Number of epoch of training')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--logflag', '-l', choices=('on', 'off'),
                    default='on', help='Writing log flag')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--net', '-n', choices=('alex', 'alexbn', 'googlenet'),
                    default='alexbn', help='Network type')
parser.add_argument('--plotflag', '-p', choices=('on', 'off'),
                    default='off', help='Accuracy plot flag')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--saveflag', '-s', choices=('on', 'off'),
                    default='off', help='Save model and optimizer flag')
args = parser.parse_args()

if args.gpu >= 0: cuda.check_cuda_available()

# Prepare dataset
print('load cifar-10 dataset')
if args.data == 'on': cifar = dh.process_data(augmentation=args.augmentation)
else: cifar = dh.load_data()

N = len(cifar['train']['x'])
N_test = len(cifar['test']['x'])
print(N, N_test)
batchsize = args.batchsize
n_epoch = args.epoch
assert N % batchsize == 0
assert N_test % batchsize == 0


# Prepare Convolution NN model
if args.net == 'alex':
    import cnn
    model = cnn.CifarCNN_2()
elif args.net == 'alexbn':
    import cnn
    model = cnn.CifarCNN_bn()
elif args.net == 'googlenet':
    import cnn_googlenet
    model = cnn_googlenet.GoogLeNet()

# GPU settings
if args.gpu >= 0:
    xp = cuda.cupy
    cuda.get_device(args.gpu).use()
    model.to_gpu()
else: xp = np

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)


# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_hdf5(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_hdf5(args.resume, optimizer)


cropwidth = 32 - model.insize
# Learning
train_ac, test_ac, train_mean_loss, test_mean_loss = [], [], [], []
stime = time.clock()
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)
    # training
    model.train = True
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N, batchsize):
        x_batch = np.reshape(cifar['train']['x'][perm[i:i + batchsize]],
                             (batchsize, 3, model.insize, model.insize))
        y_batch = cifar['train']['y'][perm[i:i + batchsize]]

        x = chainer.Variable(xp.asarray(x_batch), volatile='off')
        t = chainer.Variable(xp.asarray(y_batch), volatile='off')

        # Pass the loss function (Classifier defines it) and its arguments
        optimizer.update(model, x, t)

        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N))
    train_mean_loss.append(sum_loss / N)
    train_ac.append(sum_accuracy / N)

    # evaluation
    model.train = False

    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        val_x_batch = np.reshape(cifar['test']['x'][i:i + batchsize],
                                 (batchsize, 3, model.insize, model.insize))
        val_y_batch = cifar['test']['y'][i:i + batchsize]

        x = chainer.Variable(xp.asarray(val_x_batch), volatile='on')
        t = chainer.Variable(xp.asarray(val_y_batch), volatile='on')
        loss = model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))
    test_mean_loss.append(sum_loss / N_test)
    test_ac.append(sum_accuracy / N_test)


if args.logflag == 'on':
    import log
    etime = time.clock()
    log.write_cnn(N, N_test, batchsize, 'CNN: ' + args.net, stime, etime,
                  train_mean_loss, train_ac, test_mean_loss, test_ac, epoch,
                  LOG_FILENAME='log.txt')


if args.plotflag == 'on':
    import plot
    plot.plot_result(train_ac, test_ac, train_mean_loss, test_mean_loss,
                     savename='result_cnn.jpg')


# Save the model and the optimizer
if args.saveflag == 'on':
    serializers.save_hdf5('cifar10_alex.model', model)
    serializers.save_hdf5('cifar10_alex.state', optimizer)

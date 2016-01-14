#!/usr/bin/env python
"""Chainer example: train CNN

from __future__ import print_function

import datetime
import json
import multiprocessing
import os
import random
import sys
import threading

from PIL import Image

import six.moves.cPickle as pickle
from six.moves import queue

"""

from __future__ import print_function

import numpy as np
import six
import argparse

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers

import logging
import time
#import matplotlib
#from matplotlib import pyplot as plt

import data_cifar
# matplotlib.use('Agg')


parser = argparse.ArgumentParser(description='Example: cifar-10')

parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--logflag', '-l', choices=('on', 'off'),
                    default='off', help='Writing log flag')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--net', '-n', choices=('simple', 'parallel'),
                    default='simple', help='Network type')
parser.add_argument('--plotflag', '-p', choices=('on', 'off'),
                    default='off', help='Accuracy plot flag')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--saveflag', '-s', choices=('on', 'off'),
                    default='off', help='Save model and optimizer flag')
args = parser.parse_args()

if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np


# Prepare dataset
print('load cifar-10 dataset')
cifar = data_cifar.load_data()
cifar['train']['x'] = cifar['train']['x'].astype(np.float32)
cifar['test']['x'] = cifar['test']['x'].astype(np.float32)
cifar['train']['x'] /= 255
cifar['test']['x'] /= 255
cifar['train']['y'] = np.array(cifar['train']['y'], dtype=np.int32)
cifar['test']['y'] = np.array(cifar['test']['y'], dtype=np.int32)


N = cifar['ntraindata']
N_test = cifar['ntestdata']

n_inputs = cifar['ndim']
batchsize = 100
n_epoch = 5

assert N % batchsize == 0
assert N_test % batchsize == 0

# Prepare model
import alex_cifar
model = alex_cifar.Alex()

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()


# Setup optimizer
optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model)

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_hdf5(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_hdf5(args.resume, optimizer)


cropwidth = 32 - model.insize
train_ac, test_ac, train_mean_loss, test_mean_loss = [], [], [], []

# Learning loop
stime = time.clock()
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)
    # training
    model.train = True
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N, batchsize):
        print(i, N)
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

        x = chainer.Variable(xp.asarray(val_x_batch),
                             volatile='on')
        t = chainer.Variable(xp.asarray(val_y_batch),
                             volatile='on')


        loss = model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))
    test_mean_loss.append(sum_loss / N_test)
    test_ac.append(sum_accuracy / N_test)


if args.logflag == 'on':
    import log_cifar
    etime = time.clock()
    log_cifar.write_log(N, N_test, n_inputs, n_units, n_outputs, batchsize, 'CNN: Alex', stime, etime,
                train_mean_loss, train_ac, test_mean_loss, test_ac, epoch, LOG_FILENAME='log.txt')


if args.plotflag == 'on':
    import plot_cifar
    plot_cifar.plot_result(train_ac, test_ac, train_mean_loss, test_mean_loss, savename='result.jpg')


# Save final model
serializers.save_hdf5('cifar10_alex.model', model)
serializers.save_hdf5('cifar10_alex.state', optimizer)



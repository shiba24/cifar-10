#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on cifar

This is a minimal example to write a feed-forward net.

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

import logging
import time
import matplotlib
from matplotlib import pyplot as plt

import data_cifar
import net

matplotlib.use('Agg')


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
n_units = 5000
n_outputs = len(cifar['labels'])
batchsize = 100
n_epoch = 20


# Prepare multi-layer perceptron model, defined in net.py
if args.net == 'simple':
    model = L.Classifier(net.cifarMLP(n_inputs, n_units, n_outputs))
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy
elif args.net == 'parallel':
    cuda.check_cuda_available()
    model = L.Classifier(net.cifarMLPParallel(n_inputs, n_units, n_outputs))
    xp = cuda.cupy

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


train_ac, test_ac, train_mean_loss, test_mean_loss = [], [], [], []


# Learning loop
stime = time.clock()
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N, batchsize):
        x = chainer.Variable(xp.asarray(cifar['train']['x'][perm[i:i + batchsize]]))
        t = chainer.Variable(xp.asarray(cifar['train']['y'][perm[i:i + batchsize]]))

        # Pass the loss function (Classifier defines it) and its arguments
        optimizer.update(model, x, t)

        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N))
    train_mean_loss.append(sum_loss / N)
    train_ac.append(sum_accuracy / N)

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        x = chainer.Variable(xp.asarray(cifar['test']['x'][i:i + batchsize]),
                             volatile='on')
        t = chainer.Variable(xp.asarray(cifar['test']['y'][i:i + batchsize]),
                             volatile='on')
        loss = model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))
    test_mean_loss.append(sum_loss / N_test)
    test_ac.append(sum_accuracy / N_test)


if args.logflag == 'on':
    etime = time.clock()
    LOG_FILENAME = 'log.txt'
    logging.basicConfig(filename=LOG_FILENAME,
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s'
                        )
    logging.info(
        'New trial **************************************************\n'
        'All data: %d frames, train: %d frames / test: %d frames.\n'
        '   Inputs = %d, Units= %d, Outputs = %d, Batchsize = %d.\n'
        '   Network = %s'
        '   Total Time = %.3f sec.\n'
        '   Epoch: 1,  train mean loss=  %.5f, accuracy=  %.5f\n'
        '              test mean loss=  %.5f, accuracy=  %.5f\n'
        '   Epoch: %d, train mean loss=  %.5f, accuracy=  %.5f\n'
        '              test mean loss=  %.3f, accuracy=  %.3f\n',
        N + N_test, N, N_test,
        n_inputs, n_units, n_outputs, batchsize,
        args.net,
        etime-stime,
        train_mean_loss[0], train_ac[0],
        test_mean_loss[0], test_ac[0],
        epoch, train_mean_loss[-1], train_ac[-1],
        test_mean_loss[-1], test_ac[-1]
        )
    f = open(LOG_FILENAME, 'rt')
    try:
        body = f.read()
    finally:
        f.close()
    print('FILE:')
    print(body)


if args.plotflag == 'on':
    fig = matplotlib.pyplot.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ep = np.arange(len(train_ac)) + 1

    ax1.plot(ep, train_ac, color="blue", linewidth=2.5, linestyle="-", label="Train")
    ax1.plot(ep, test_ac, color="red",  linewidth=2.5, linestyle="-", label="Test")

    ax2.plot(ep, train_mean_loss, color="blue", linewidth=2.5, linestyle="-", label="Train")
    ax2.plot(ep, test_mean_loss, color="red",  linewidth=2.5, linestyle="-", label="Test")

    ax1.set_title("Accuracy")
    ax2.set_title("Mean Loss")

    ax1.set_xlabel("epoch")
    ax2.set_xlabel("epoch")
    fig.tight_layout()
    matplotlib.pyplot.legend(loc='upper right')

    matplotlib.pyplot.savefig('result.jpg')


# Save the model and the optimizer
if args.saveflag == 'on':
    print('save the model')
    serializers.save_hdf5('cifar10.model', model)
    print('save the optimizer')
    serializers.save_hdf5('cifar10.state', optimizer)

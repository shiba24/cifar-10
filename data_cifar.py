import os
import cPickle

import numpy as np
import six
from six.moves.urllib import request

dataurl = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
filepath = 'cifar-10-batches-py/'
flist = ['data_batch_1', 'data_batch_2', 'data_batch_3',
         'data_batch_4', 'data_batch_5', 'test_batch', 'batches.meta']


def download_data():
    """Downloading data from URL.
    Need to rezip manually.
    """
    print('Downloading {:s}...'.format(filename))
    request.urlretrieve(dataurl)
    print('Done')
    print('Please unzip files. command is:')
    print('gzip -d cifar-10-python.tar.gz')
    print('tar -xf cifar-10-python.tar')


def read(fname):
    print 'open batch file'
    fo = open(fname, 'rb')
    dic = cPickle.load(fo)
    fo.close()
    return dic


def convert_train(ndata, ndim):
    """Converting all the training data.
    Take some time and memory.
    If not possible for your environment,
    try "load_batch(n)" function.
    """
    print 'Converting training data ... '
    x = np.zeros([ndata, ndim])
    y = np.zeros([ndata])
    for i in range(0, len(flist) - 2):
        batchn = filepath + flist[i]
        temp = read(batchn)
        x[i * 10000:(i + 1) * 10000] = temp['data']
        y[i * 10000:(i + 1) * 10000] = temp['labels']
    return x, y


def save_pkl(data, savename='cifar.pkl'):
    print('Saving data .pkl...')
    with open(savename, 'wb') as output:
        six.moves.cPickle.dump(data, output, -1)
    print('Done')


def set_data():
    """Set data for all the training and test, at one time.
    Take some time and memory.
    If not possible for your environment,
    try "set_batch_data()" function.
    """
    if not os.path.exists(filepath):
        download_data()
    metadata = read(filepath + flist[-1])
    ndata = metadata['num_cases_per_batch']
    ndim = metadata['num_vis']

    data, train, test = {}, {}, {}
    data['labels'] = metadata['label_names']
    data['ntraindata'] = metadata['num_cases_per_batch'] * (len(flist) - 2)
    data['ntestdata'] = metadata['num_cases_per_batch']
    data['ndim'] = metadata['num_vis']

    train['x'], train['y'] = convert_train(data['ntraindata'], data['ndim'])

    testdata = read(filepath + flist[-2])
    test['x'] = testdata['data']
    test['y'] = testdata['labels']

    data['train'], data['test'] = train, test
    save_pkl(data)


def set_batch_data():
    """Set data for either one training or test at one time.
    For less memory consumption.
    """
    for n in range(0,6):
        d = read(filepath + flist[n])
        metadata = read(filepath + flist[-1])
        ndata = metadata['num_cases_per_batch']
        ndim = metadata['num_vis']

        data, trts = {}, {}
        data['labels'] = metadata['label_names']
        data['ntraindata'] = metadata['num_cases_per_batch'] * (len(flist) - 2)
        data['ntestdata'] = metadata['num_cases_per_batch']
        data['ndim'] = metadata['num_vis']
        trts['x'], trts['y'] = d['data'], d['labels']
        trtsflag = ['train', 'train', 'train', 'train', 'train', 'test']

        data['flag'] = trtsflag[n]
        data[trtsflag[n]] = trts
        save_pkl(data, savename=flist[n]+'.pkl')


def load_data():
    """Load all the data for the training and test, at one time.
    Take some time and memory.
    If not possible for your environment,
    try "load_batch(n)" function.
    """
    print 'Loadng all the file one time......'
    if not os.path.exists('cifar.pkl'):
        set_data()
    with open('cifar.pkl', 'rb') as cifar_pickle:
        data = six.moves.cPickle.load(cifar_pickle)
    return data


def load_batch(n):
    """Load only one data for training or test, at one time.
    For less memory consumption.
    """
    print 'Loadng one batch...'
    batchfilename = flist[n - 1] + '.pkl'
    if not os.path.exists(batchfilename):
        set_batch_data()
    with open(batchfilename, 'rb') as cifar_pickle:
        data = six.moves.cPickle.load(cifar_pickle)
    return data


if __name__ == "__main__":
    set_data()


"""
MEMO

{'num_cases_per_batch': 10000,
'label_names':
['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
'num_vis': 3072}
"""

"""
MEMO

data
    a 10000x3072 numpy array of uint8s
    Each row of the array stores a 32x32 colour image.
    The first 1024 entries contain the red channel values,
    the next 1024 the green, and the final 1024 the blue.
    The image is stored in row-major order,
    so that the first 32 entries of the array are
    the red channel values of the first row of the image.
labels
    a list of 10000 numbers in the range 0-9.
    The number at index i indicates the label of
    the ith image in the array data.
"""

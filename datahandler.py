import numpy as np
from numpy import linalg as la
import os
import data
import six
from tqdm import tqdm


def normalize(data, M=None, Sd=None):
    """ Normalize data """
    print('Data normalization......')
    if M == None:
        M = np.mean(data, axis=0)        # mean
    if Sd == None:
        Sd = np.std(data, axis=0)        # Std

    stmat = np.zeros([len(Sd), len(Sd)])
    for i in range(0, len(Sd)):
        stmat[i][i] = Sd[i]
    S_inv = la.inv(np.matrix(stmat))
    data_n = S_inv.dot((np.matrix(data - M)).T)
    data_n = np.array(data_n.T, dtype=np.float32)
    return data_n, M, Sd


def pad_addnoise(data_x, data_y, mean=0.0, sd=1.0, mixratio=1.0, noiseratio=0.3):
    """ Data padding: add noise ~ N(mean, sd) """
    print('Data padding: adding noise to data......')
    col, row = np.int(len(data_x) * mixratio), len(data_x[0])
    noise = np.random.normal(mean, sd, (col, row)) * np.sqrt(noiseratio)
    noised = np.array(data_x[0:col] * np.sqrt(1 - noiseratio) + noise, dtype=np.float32)
    noised_x = np.append(data_x, noised, axis=0)
    noised_y = np.append(data_y, data_y[0:col], axis=0)
    return noised_x, noised_y


def pad_rightleft(data_x, data_y, mixratio=1.0, ch=3):
    """ Data padding: rightside left """
    print('Data padding: rightside left......')
    col, size = np.int(len(data_x) * mixratio), len(data_x[0]) / ch
    height, width = 32, 32

    rl = data_x.copy()
    for i in tqdm(range(0, ch)):
        for j in range(0, height):
            r = data_x[:, i * size + j * height:i * size + j * height + width]
            rl[:, i * size + j * height:i * size + j * height + width] = np.fliplr(r)
    rl_x = np.append(data_x, rl[0:col], axis=0)
    rl_y = np.append(data_y, data_y[0:col], axis=0)
    return rl_x, rl_y


def crop_data(data_x, data_y, imagesize=32, insize=24, stride=4, ch=3):
    print('Data cropping: images x 3 x 3......')
    ratio = (imagesize - insize) / stride + 1
    imagemat, labelvec = data_x, data_y
    nimage = len(imagemat)

    cropped_x = np.zeros([nimage * ratio ** 2, insize ** 2 * ch])
    cropped_y = np.zeros(nimage * ratio ** 2)
    for i in tqdm(range(0, nimage)):
        for c in range(0,ch):
            imagemat = data_x[i][c * imagesize ** 2:(c + 1) * imagesize ** 2]
            for j in range(0, ratio):
                xind = j * stride
                for k in range(0, ratio):
                    yind = k * stride
                    cropped_x[i * ratio**2 + j * ratio + k, c * insize ** 2:(c + 1) * insize ** 2] = np.reshape(
                        np.reshape(imagemat, (imagesize, imagesize))[yind:yind + insize, xind:xind + insize],
                        (1, insize ** 2))
        cropped_y[i * ratio ** 2: (i + 1) * ratio ** 2] = np.ones(ratio ** 2).astype(np.int32) * labelvec[i]

    data_x = cropped_x.astype(np.float32)
    data_y = np.array(cropped_y, dtype=np.int32)
    return data_x, data_y


def process_data(augmentation=2):
    cifar = load_data()
    cifar['train']['x'], m, sd = normalize(cifar['train']['x'])
    cifar['test']['x'], m, sd = normalize(cifar['test']['x'], M=m, Sd=sd)
    if augmentation > 0:
        cifar['train']['x'], cifar['train']['y'] = pad_rightleft(cifar['train']['x'], cifar['train']['y'],
                                                                 mixratio=augmentation)
    if augmentation > 1.0:
        cifar['train']['x'], cifar['train']['y'] = pad_addnoise(cifar['train']['x'], cifar['train']['y'],
                                                                mixratio=augmentation - 1.0)
#    data.save_pkl(cifar, savename='cifar_processed.pkl')
    return cifar


def load_processed_data():
    if not os.path.exists('cifar_processed.pkl'):
        cifar = process_data()
        return cifar
    else:
        with open('cifar_processed.pkl', 'rb') as cifar_pickle:
            data = six.moves.cPickle.load(cifar_pickle)
    return data


def load_data():
    cifar = data.load_data()
    cifar['train']['x'] = cifar['train']['x'].astype(np.float32)
    cifar['test']['x'] = cifar['test']['x'].astype(np.float32)
    cifar['train']['x'] /= 255
    cifar['test']['x'] /= 255
    cifar['train']['y'] = np.array(cifar['train']['y'], dtype=np.int32)
    cifar['test']['y'] = np.array(cifar['test']['y'], dtype=np.int32)
    return cifar


"""
def pad_updown(data):
    return data

def pad_crop(data):
    return data

def load_processed_data():
    if not os.path.exists('cifar_processed.pkl'):
        cifar = process_data()
        return cifar
    else:
        with open('cifar_processed.pkl', 'rb') as cifar_pickle:
            data = six.moves.cPickle.load(cifar_pickle)
        return data

    imagemat = np.reshape(imagemat, (nimage, imagesize, imagesize, ch))

#    imagemat = np.transpose(np.reshape(imagemat, (nimage, imagesize, imagesize, ch)), (0, 3, 1, 2))
                    cropped_x[i + j * ratio + k, c * insize ** 2:(c + 1) * insize ** 2] = np.reshape(
                        imagemat[xind:xind + insize, yind:yind + insize],
                                                      (1, insize ** 2 * ch))

"""

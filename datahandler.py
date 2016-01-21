import numpy as np
from numpy import linalg as la
import os
import data


def normalize(data):
    """ Normalize data """
    print('Data normalization......')
    M = np.mean(data, axis=0)        # mean
    Sd = np.std(data, axis=0)        # Std
    stmat = np.zeros([len(Sd), len(Sd)])
    for i in range(0, len(Sd)):
        stmat[i][i] = Sd[i]
    S_inv = la.inv(np.matrix(stmat))
    data_n = S_inv.dot((np.matrix(data - M)).T)
    data_n = np.array(data_n.T, dtype=np.float32)
    return data_n


def pad_addnoise(data_x, data_y, mean=0.0, sd=1.0, noiseratio=0.5, mixratio=0.2):
    """ Add noise ~ N(mean, sd) """
    print('Adding noise to data......')
    col, row = np.int(len(data_x) * mixratio), len(data_x[0])
    noise = np.random.normal(mean, sd, (col, row)) * np.sqrt(noiseratio)
    noised = np.array(data_x[0:col] * np.sqrt(1 - noiseratio) + noise, dtype=np.float32)
    noised_x = np.append(data_x, noised, axis=0)
    noised_y = np.append(data_y, data_y[0:col], axis=0)
    return noised_x, noised_y


def pad_rightleft(data_x, data_y, mixratio=1.0, ch=3):
    """ Rightside left """
    print('Rightside to left......')
    col, row = np.int(len(data_x) * mixratio), len(data_x[0]) / ch

    rl = data_x.copy()
    for i in range(0, ch):
        r = data_x[i * row:(i + 1) * row]
        rl[i * row:(i+1) * row] = np.fliplr(r)
    rl_x = np.append(data_x, rl[0:col], axis=0)
    rl_y = np.append(data_y, data_y[0:col], axis=0)
    return rl_x, rl_y


def process_data():
    cifar = load_data()
    cifar['train']['x'] = normalize(cifar['train']['x'])
    cifar['test']['x'] = normalize(cifar['test']['x'])
    cifar['train']['x'], cifar['train']['y'] =  pad_rightleft(cifar['train']['x'], cifar['train']['y'])
    cifar['train']['x'], cifar['train']['y'] =  pad_addnoise(cifar['train']['x'], cifar['train']['y'])
#    data.save_pkl(cifar, savename='cifar_processed.pkl')
    return cifar


def load_processed_data():
    cifar = process_data()
    return cifar


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
"""

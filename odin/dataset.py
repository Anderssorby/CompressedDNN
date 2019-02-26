#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import glob
import logging
import os
import sys
from abc import abstractmethod
from subprocess import check_call

import numpy as np
from scipy import linalg
from six.moves import cPickle as pickle
from skimage.io import imsave

from odin.utils.transformer import Transformer
from odin.utils.generate import *

data_dir = "data"


def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()

    return data


def load_dataset(name, options={}):
    # type: (str) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    dataset = datasets[name](**options)

    return dataset.load()


def preprocessing(data):
    mean = np.mean(data, axis=0)
    mdata = data - mean
    sigma = np.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = linalg.svd(sigma)
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S))), U.T)
    whiten = np.dot(mdata, components.T)

    return components, mean, whiten


def download_and_unwrap_tarball(source):
    logging.info("Downloading from %s" % source)
    check_call(["wget", source])
    name = source.split("/")[-1]
    check_call(["tar", "zxvf", name])
    check_call(["rm", "-rf", name])


class Dataset:
    """
    Superclass of data sets
    """

    def __init__(self,
                 whitening=False,
                 prompt=True, **kwargs):
        self.path = os.path.join(data_dir, self.name)
        self.whitening = whitening
        self.prompt = prompt
        self.args = kwargs

    def _extract_train(self):
        # type: () -> (list, list)
        pass

    def _extract_test(self):
        # type: () -> (list, list)
        pass

    def _download(self):
        download_and_unwrap_tarball(self.source)

    def _prepare_and_save(self):
        logging.info("Preparing dataset.")

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # prepare training dataset
        data, labels = self._extract_train()

        np.save(os.path.join(self.path, 'train_data'),
                np.asarray(data, dtype=np.float32))
        np.save(os.path.join(self.path, 'train_labels'),
                np.asarray(labels, dtype=np.int32))

        data, labels = self._extract_test()

        np.save(os.path.join(self.path, 'test_data'), data)
        np.save(os.path.join(self.path, 'test_labels'), labels)

    def load(self):
        if not os.path.isdir(self.path):
            try:
                if self.prompt:
                    answer = input("Do you want to download %s? [Y/n]" % self.name).lower()
                    if answer == "n":
                        exit(0)

                self._download()

                self._prepare_and_save()
            except Exception:
                print("Cleaning up broken dir ".join(self.path))
                # check_call(["rm", "-rf", self.path])

        train_data = np.load(os.path.join(self.path, 'train_data.npy'))
        train_labels = np.load(os.path.join(self.path, 'train_labels.npy'))
        test_data = np.load(os.path.join(self.path, 'test_data.npy'))
        test_labels = np.load(os.path.join(self.path, 'test_labels.npy'))

        return train_data, train_labels, test_data, test_labels


class Cifar10(Dataset):
    source = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    name = "cifar10"

    def _extract_train(self):

        trans = Transformer()
        data = np.zeros((50000, 3 * 32 * 32), dtype=np.float)
        labels = []
        for i, data_fn in enumerate(
                sorted(glob.glob('cifar-10-batches-py/data_batch*'))):
            batch = unpickle(data_fn)
            data[i * 10000:(i + 1) * 10000] = batch['data']
            labels.extend(batch['labels'])

        if self.whitening:
            components, mean, data = preprocessing(data)
            np.save(os.path.join(self.path, "components"), components)
            np.save(os.path.join(self.path, "mean"), mean)

        data = data.reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))
        labels = np.asarray(labels, dtype=np.int32)
        training_data = []
        training_labels = []
        for d, l in zip(data, labels):
            imgs = trans(d)
            for img in imgs:
                training_data.append(img)
                training_labels.append(l)

        # saving training dataset
        for i in range(50000):
            d = data[i]
            d -= d.min()
            d /= d.max()
            d = (d * 255).astype(np.uint8)
            imsave(os.path.join(self.path, 'train_{}.png'.format(i)), d)

        return training_data, training_labels

    def _extract_test(self):
        test = unpickle('cifar-10-batches-py/test_batch')
        data = np.asarray(test['data'], dtype=np.float)
        data = data.reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))
        labels = np.asarray(test['labels'], dtype=np.int32)

        for i in range(10000):
            d = data[i]
            d -= d.min()
            d /= d.max()
            d = (d * 255).astype(np.uint8)
            imsave(os.path.join(self.path, 'test_{}.png'.format(i)), d)

        return data, labels


class MNIST(Dataset):
    def _extract_test(self):
        unpickle("train-images-idx3-ubyte.gz")

    def _extract_train(self):
        pass

    name = "mnist"

    def _download(self):
        sources = [
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
        ]
        for i, source in enumerate(sources):
            print("Downloading %d of 4" % (i+1))
            download_and_unwrap_tarball(source)

    def load(self):
        import chainer
        train, test = chainer.datasets.get_mnist()
        return train, test


class PTBWords(Dataset):

    name = "ptb_words"

    def load(self):
        import chainer
        train, val, test = chainer.datasets.get_ptb_words()
        return train, val, test


class Mini(Dataset):
    name = "mini"

    def load(self):
        file = os.path.join(data_dir, "mini.npy")
        if not os.path.isfile(file):
            train, test = generate_dataset()

            if not os.path.isdir(data_dir):
                os.mkdir(data_dir)

            np.save(file, (train, test))
        else:
            train, test = np.load(file)

        return train, test


datasets = {
    "cifar10": Cifar10,
    "mnist": MNIST,
    "ptb_words": PTBWords,
    "mini": Mini
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='cifar-10', choices=datasets.keys())
    parser.add_argument('--whitening', action='store_true', default=False)
    args = parser.parse_args()

    load_dataset(name=args.name, whitening=args.whitening)
    # print(args)

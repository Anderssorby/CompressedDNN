# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from scipy import linalg as LA

from odin import results_dir
from .base import ComputationInterface
from glob import glob


# noinspection PyUnresolvedReferences
class Chainer(ComputationInterface):
    def __init__(self):
        super(Chainer, self).__init__()

        self.cuda = None
        self.xp = None

        self.using_gpu = False

    def update_args(self, args):
        super(Chainer, self).update_args(args)

        if self.args.gpu >= 0:
            print("Getting GPU")
            from chainer import cuda
            self.cuda = cuda
            cuda.get_device(self.args.gpu).use()
            self.xp = cuda.cupy
            self.using_gpu = True
        else:

            import numpy as np
            self.xp = np

    def load_group(self, group_name, model_wrapper):
        path = os.path.join(results_dir, model_wrapper.model_name, group_name, "*.npy")
        group = {}
        for f in glob(path):
            element = self.xp.load(f)

            name = f.split("/")[-1].split(".")[0]
            group[name] = element
        return group

    def store_elements(self, elements, group_name, model_wrapper):
        path = os.path.join(results_dir, model_wrapper.model_name, group_name)
        if not os.path.isdir(path):
            os.makedirs(path)

        for key in elements:
            self.xp.save(os.path.join(path, key), elements[key])

    def calc_inter_layer_covariance(self, model_wrapper, use_training_data=False):

        model = model_wrapper.model

        if self.using_gpu:
            model.to_gpu()

        xp = self.xp

        train, test = model_wrapper.dataset

        data = train if use_training_data else test

        data_size = len(data)
        batch_size = 1000

        model.train = False

        perm = xp.random.permutation(data_size)

        full_mean = []
        cov = []
        test_num = 0

        for batch in range(0, data_size, batch_size):
            test_num += 1
            ind_tmp = perm[batch:batch + batch_size]

            x = xp.asarray(data[ind_tmp][0], dtype=xp.float32)

            layer_outputs = model.predictor(x, multi_layer=True)
            for i, layer_out in enumerate(layer_outputs):
                tmp_mean = xp.mean(layer_out.data, axis=0)

                # part_size = layer_out.data.shape
                tmp_cov = xp.dot(layer_out.data, layer_out.data.T)
                if batch == 0:
                    full_mean.append(tmp_mean)

                    cov.append(tmp_cov)
                else:
                    full_mean[i] += tmp_mean
                    cov[i] += tmp_cov

        eigen_values = []
        for jj, layer_mean in enumerate(full_mean):
            # mean_cross = xp.outer(layer_mean, layer_mean)
            # mean_cross = mean_cross / layer_mean.shape[0]
            cov[jj] = cov[jj] / test_num
            if self.using_gpu:
                tmp_cov_ma = self.cuda.to_cpu(cov[jj])
            else:
                tmp_cov_ma = cov[jj]
            eigs = LA.eigvals(tmp_cov_ma)
            eigen_values.append(eigs)

        # Saving
        self.store_elements({
            "cov": cov,
            "eigen_values": eigen_values,
        }, group_name="inter_layer_covariance", model_wrapper=model_wrapper)

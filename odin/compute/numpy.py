# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from progress.bar import ChargingBar
from scipy import linalg as LA

from .base import ComputationInterface


# noinspection PyUnresolvedReferences
class Numpy(ComputationInterface):
    def __init__(self):
        super(Numpy, self).__init__()

        self.cuda = None

        self.using_gpu = False

    def update_args(self, args):
        super(Numpy, self).update_args(args)

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

    def calc_block_inter_layer_covariance(self, model_wrapper, use_training_data=True, batch_size=100, **options):
        model = model_wrapper.model
        is_chainer = model_wrapper.model_type == "chainer"

        if self.using_gpu and is_chainer:
            model.to_gpu()

        xp = self.xp

        train, test = model_wrapper.dataset

        data_x = train if use_training_data else test

        if is_chainer:
            data_x = xp.moveaxis(data_x, -1, 0)[0]
        else:
            data_x = data_x[0]

        data_x = xp.stack(data_x, axis=0)

        data_size = len(data_x)

        if batch_size > 0:
            perm = xp.random.permutation(data_size)
            data_x = data_x[perm[0:batch_size]]
            data_size = batch_size

        compute_batch_size = 100

        if is_chainer:
            model.train = False

        # perm = xp.random.permutation(data_size)

        full_mean = []
        cov = []
        test_num = 0
        bar = ChargingBar("Calculating inter layer covariance", max=data_size)
        for batch in range(0, data_size, compute_batch_size):
            test_num += 1
            # ind_tmp = perm[batch:batch + batch_size] xp.asarray(, dtype=xp.float32)

            x = data_x[batch:batch + compute_batch_size]

            layer_outputs = model_wrapper.get_layer_outputs(x)
            for i, layer_out in enumerate(layer_outputs):
                if is_chainer:
                    layer_out = layer_out.data
                # tmp_mean = xp.mean(layer_out, axis=0)

                # tmp_cov = xp.einsum("a,b->ab", layer_out.ravel(), layer_out.ravel())

                tmp_cov = None
                for fxi in layer_out:
                    fxi = fxi.ravel()
                    dot = xp.outer(fxi, fxi)
                    if tmp_cov is None:
                        tmp_cov = dot
                    else:
                        tmp_cov += dot
                    bar.next()

                if batch == 0:
                    full_mean.append(tmp_mean)

                    cov.append(tmp_cov)
                else:
                    # full_mean[i] += tmp_mean
                    cov[i] += tmp_cov
            logging.debug("Computed covariance for batch %d of size %d" % (batch, compute_batch_size))
        bar.finish()

        bar = ChargingBar("Calculating eigen values", max=len(cov))

        eigen_values = []
        for j in range(len(cov)):
            cov[j] = cov[j] / test_num
            if self.using_gpu:
                tmp_cov_ma = self.cuda.to_cpu(cov[j])
            else:
                tmp_cov_ma = cov[j]
            eigs = LA.eigvals(tmp_cov_ma)
            eigen_values.append(eigs)
            bar.next()
        bar.finish()

        # Saving
        self.store_elements({
            "cov": cov,
            "eigen_values": eigen_values,
        }, group_name="inter_layer_covariance", model_wrapper=model_wrapper)

    def calc_inter_layer_covariance(self, model_wrapper, use_training_data=True, batch_size=100, **options):

        model = model_wrapper.model
        is_chainer = model_wrapper.model_type == "chainer"

        if self.using_gpu and is_chainer:
            model.to_gpu()

        xp = self.xp

        train, test = model_wrapper.dataset

        data_x = train if use_training_data else test

        if is_chainer:
            data_x = xp.moveaxis(data_x, -1, 0)[0]
        else:
            data_x = data_x[0]

        data_x = xp.stack(data_x, axis=0)

        data_size = len(data_x)

        if batch_size and batch_size > 0:
            perm = xp.random.permutation(data_size)
            data_x = data_x[perm[0:batch_size]]
            data_size = batch_size

        compute_batch_size = 100

        if is_chainer:
            model.train = False

        # perm = xp.random.permutation(data_size)

        # full_mean = []
        cov = []
        test_num = 0
        bar = ChargingBar("Calculating inter layer covariance", max=data_size)
        for batch in range(0, data_size, compute_batch_size):
            test_num += 1
            # ind_tmp = perm[batch:batch + batch_size] xp.asarray(, dtype=xp.float32)

            x = data_x[batch:batch + compute_batch_size]

            layer_outputs = model_wrapper.get_layer_outputs(x)
            for i, layer_out in enumerate(layer_outputs):
                if is_chainer:
                    layer_out = layer_out.data
                # tmp_mean = xp.mean(layer_out, axis=0)

                # tmp_cov = xp.einsum("a,b->ab", layer_out.ravel(), layer_out.ravel())

                tmp_cov = None
                for fxi in layer_out:
                    fxi = fxi.ravel()
                    dot = xp.outer(fxi, fxi)
                    if tmp_cov is None:
                        tmp_cov = dot
                    else:
                        tmp_cov += dot
                    bar.next()

                if batch == 0:
                    # full_mean.append(tmp_mean)

                    cov.append(tmp_cov)
                else:
                    # full_mean[i] += tmp_mean
                    cov[i] += tmp_cov
            logging.debug("Computed covariance for batch %d of size %d" % (batch, compute_batch_size))
        bar.finish()

        bar = ChargingBar("Calculating eigen values", max=len(cov))

        eigen_values = []
        for j in range(len(cov)):
            cov[j] = cov[j] / test_num
            if self.using_gpu:
                tmp_cov_ma = self.cuda.to_cpu(cov[j])
            else:
                tmp_cov_ma = cov[j]
            eigs = LA.eigvals(tmp_cov_ma)
            eigen_values.append(eigs)
            bar.next()
        bar.finish()

        # Saving
        self.store_elements({
            "cov": cov,
            "eigen_values": eigen_values,
        }, group_name="inter_layer_covariance", model_wrapper=model_wrapper)

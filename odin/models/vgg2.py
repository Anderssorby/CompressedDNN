#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer.functions as F
import chainer.links as L
import chainer.serializers
import logging
import numpy as np
import os
import shutil
import six
from chainer import Variable
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from multiprocessing import Process
from multiprocessing import Queue

import odin.plot
from odin.utils.transformer import Transformer
from .base import ChainerModelWrapper


class VGG2Wrapper(ChainerModelWrapper):
    model_name = "VGG2"
    dataset_name = "cifar-10"

    def construct(self):
        model = VGG2()
        return model

    def train(self, x_train=None, y_train=None, **options):
        optimizer = get_model_optimizer(self.model, args=options)
        # create model and optimizer
        # model, optimizer = get_model_optimizer(args)
        # dataset = load_dataset(args.datadir)
        tr_data, tr_labels, te_data, te_labels = self.dataset
        epochs = options.get("epoch", default=10)
        validate_freq = options.get("validate_freq", default=5)
        lr_decay_freq = options.get("lr_decay_freq", default=1)
        lr_decay_ratio = options.get("lr_decay_ratio", default=0.1)

        args = {}

        # learning loop
        for epoch in range(1, epochs + 1):
            logging.info('learning rate:{}'.format(optimizer.lr))

            one_epoch(args, self.model, optimizer, tr_data, tr_labels, epoch, train=True)

            if epoch == 1 or epoch % validate_freq == 0:
                one_epoch(args, self.model, optimizer, te_data, te_labels, epoch, train=False)

            if args.opt == 'MomentumSGD' and epoch % lr_decay_freq == 0:
                optimizer.lr *= lr_decay_ratio
        one_epoch(model=self.model, optimizer=optimizer)


class VGG2(chainer.Chain):
    def __init__(self, wbase=64):
        super(VGG2, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, pad=1),
            bn1_1=L.BatchNormalization(64),
            conv1_2=L.Convolution2D(64, 64, 3, pad=1),
            bn1_2=L.BatchNormalization(64),

            conv2_1=L.Convolution2D(64, 128, 3, pad=1),
            bn2_1=L.BatchNormalization(128),
            conv2_2=L.Convolution2D(128, 128, 3, pad=1),
            bn2_2=L.BatchNormalization(128),

            conv3_1=L.Convolution2D(128, 256, 3, pad=1),
            bn3_1=L.BatchNormalization(256),
            conv3_2=L.Convolution2D(256, 256, 3, pad=1),
            bn3_2=L.BatchNormalization(256),
            conv3_3=L.Convolution2D(256, 256, 3, pad=1),
            bn3_3=L.BatchNormalization(256),
            conv3_4=L.Convolution2D(256, 256, 3, pad=1),
            bn3_4=L.BatchNormalization(256),

            fc4=L.Linear(2304, 1024),
            fc5=L.Linear(1024, 1024),
            fc6=L.Linear(1024, 10),
        )
        self.train = True

    def __call__(self, x, t, b_Anal=False):
        print('kita')
        h = F.relu(self.bn1_1(self.conv1_1(x), test=not self.train))
        if b_Anal:
            i = 1
            hout = [('h{}'.format(i), h)]
        h = F.relu(self.bn1_2(self.conv1_2(h), test=not self.train))
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i), h)]
        h = F.max_pooling_2d(h, 2, 2)
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i), h)]
        h = F.dropout(h, ratio=0.25, train=self.train)

        h = F.relu(self.bn2_1(self.conv2_1(h), test=not self.train))
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i), h)]
        h = F.relu(self.bn2_2(self.conv2_2(h), test=not self.train))
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i), h)]
        h = F.max_pooling_2d(h, 2, 2)
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i), h)]
        h = F.dropout(h, ratio=0.25, train=self.train)

        h = F.relu(self.bn3_1(self.conv3_1(h), test=not self.train))
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i), h)]
        h = F.relu(self.bn3_2(self.conv3_2(h), test=not self.train))
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i), h)]
        h = F.relu(self.bn3_3(self.conv3_3(h), test=not self.train))
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i), h)]
        h = F.relu(self.bn3_4(self.conv3_4(h), test=not self.train))
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i), h)]
        h = F.max_pooling_2d(h, 2, 2)
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i), h)]
        h = F.dropout(h, ratio=0.25, train=self.train)

        h = F.dropout(F.relu(self.fc4(h)), ratio=0.5, train=self.train)
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i), h)]
        h = F.dropout(F.relu(self.fc5(h)), ratio=0.5, train=self.train)
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i), h)]
        h = self.fc6(h)
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i), h)]

        if b_Anal:
            return hout

        self.pred = F.softmax(h)
        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(self.pred, t)

        if self.train:
            return self.loss
        else:
            return self.pred


def get_model_optimizer(model, args):
    model_fn = os.path.basename(args.model)
    # model = VGG2(wbase=args.wbase)

    dst = '%s/%s' % (args.result_dir, model_fn)
    if not os.path.exists(dst):
        shutil.copy(args.model, dst)

    dst = '%s/%s' % (args.result_dir, os.path.basename(__file__))
    if not os.path.exists(dst):
        shutil.copy(__file__, dst)

    # prepare model
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # prepare optimizer
    if 'opt' in args:
        # prepare optimizer
        if args.opt == 'MomentumSGD':
            optimizer = optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
        elif args.opt == 'Adam':
            optimizer = optimizers.Adam(alpha=args.alpha)
        elif args.opt == 'AdaGrad':
            optimizer = optimizers.AdaGrad(lr=args.lr)
        else:
            raise Exception('No optimizer is selected')

        optimizer.setup(model)

        if args.opt == 'MomentumSGD':
            optimizer.add_hook(
                chainer.optimizer.WeightDecay(args.weight_decay))
        return optimizer
    else:
        print('No optimizer generated.')


def augmentation(args, aug_queue, data, label, train):
    trans = Transformer(args)
    # np.random.seed(int(time.time()))
    perm = np.random.permutation(data.shape[0])
    if train:
        for i in six.moves.range(0, data.shape[0], args.batchsize):
            chosen_ids = perm[i:i + args.batchsize]
            x = np.asarray(data[chosen_ids], dtype=np.float32)
            x = x.transpose((0, 3, 1, 2))
            t = np.asarray(label[chosen_ids], dtype=np.int32)
            aug_queue.put((x, t))
    else:
        for i in six.moves.range(data.shape[0]):
            aug = trans(data[i])
            x = np.asarray(aug, dtype=np.float32).transpose((0, 3, 1, 2))
            t = np.asarray(np.repeat(label[i], len(aug)), dtype=np.int32)
            aug_queue.put((x, t))
    aug_queue.put(None)
    return


def one_epoch(args, model, optimizer, data, label, epoch, train):
    model.train = train
    xp = cuda.cupy if args.gpu >= 0 else np

    # for parallel augmentation
    aug_queue = Queue()
    aug_worker = Process(target=augmentation,
                         args=(args, aug_queue, data, label, train))
    aug_worker.start()
    logging.info('data loading started')

    sum_accuracy = 0
    sum_loss = 0
    num = 0
    while True:
        datum = aug_queue.get()
        if datum is None:
            break
        x, t = datum

        volatile = 'off' if train else 'on'
        x = Variable(xp.asarray(x), volatile=volatile)
        t = Variable(xp.asarray(t), volatile=volatile)

        if train:
            optimizer.update(model, x, t)
            if epoch == 1 and num == 0:
                with open('{}/graph.dot'.format(args.result_dir), 'w') as o:
                    g = computational_graph.build_computational_graph(
                        (model.loss,), remove_split=True)
                    o.write(g.dump())
            sum_loss += float(model.loss.data) * len(t.data)
            sum_accuracy += float(model.accuracy.data) * len(t.data)
            num += t.data.shape[0]
            logging.info('{:05d}/{:05d}\tloss:{:.3f}\tacc:{:.3f}'.format(
                num, data.shape[0], sum_loss / num, sum_accuracy / num))
        else:
            pred = model(x, t).data
            pred = pred.mean(axis=0)
            acc = int(pred.argmax() == t.data[0])
            sum_accuracy += acc
            num += 1
            logging.info('{:05d}/{:05d}\tacc:{:.3f}'.format(
                num, data.shape[0], sum_accuracy / num))

        del x, t

    if train and (epoch == 1 or epoch % args.snapshot == 0 or epoch == args.epoch):
        model_fn = '{}/epoch-{}.model'.format(args.result_dir, epoch)
        opt_fn = '{}/epoch-{}.state'.format(args.result_dir, epoch)
        serializers.save_hdf5(model_fn, model)
        serializers.save_hdf5(opt_fn, optimizer)

    if train:
        logging.info('epoch:{}\ttrain loss:{}\ttrain accuracy:{}'.format(
            epoch, sum_loss / data.shape[0], sum_accuracy / data.shape[0]))
    else:
        logging.info('epoch:{}\ttest loss:{}\ttest accuracy:{}'.format(
            epoch, sum_loss / data.shape[0], sum_accuracy / data.shape[0]))

    odin.plot.draw_loss_curve('{}/log.txt'.format(args.result_dir),
                              '{}/log.png'.format(args.result_dir), epoch)

    aug_worker.join()
    logging.info('data loading finished')

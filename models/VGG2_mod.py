#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L

import VGG2 as vgg2

class VGG_mod(vgg2.VGG):

    def __call__(self, x, t, b_Anal = False):
        h = F.relu(self.bn1_1(self.conv1_1(x), test=not self.train))
        if b_Anal:
            i = 1
            hout = [('h{}'.format(i),h)]
        h = F.relu(self.bn1_2(self.conv1_2(h), test=not self.train))
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i),h)]
        h = F.max_pooling_2d(h, 2, 2)
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i),h)]
        h = F.dropout(h, ratio=0.25, train=self.train)

        h = F.relu(self.bn2_1(self.conv2_1(h), test=not self.train))
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i),h)]
        h = F.relu(self.bn2_2(self.conv2_2(h), test=not self.train))
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i),h)]
        h = F.max_pooling_2d(h, 2, 2)
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i),h)]
        h = F.dropout(h, ratio=0.25, train=self.train)

        h = F.relu(self.bn3_1(self.conv3_1(h), test=not self.train))
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i),h)]
        h = F.relu(self.bn3_2(self.conv3_2(h), test=not self.train))
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i),h)]
        h = F.relu(self.bn3_3(self.conv3_3(h), test=not self.train))
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i),h)]
        h = F.relu(self.bn3_4(self.conv3_4(h), test=not self.train))
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i),h)]
        h = F.max_pooling_2d(h, 2, 2)
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i),h)]
        h = F.dropout(h, ratio=0.25, train=self.train)

        h = F.dropout(F.relu(self.fc4(h)), ratio=0.5, train=self.train)
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i),h)]
        h = F.dropout(F.relu(self.fc5(h)), ratio=0.5, train=self.train)
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i),h)]
        h = self.fc6(h)
        if b_Anal:
            i += 1
            hout += [('h{}'.format(i),h)]

        if b_Anal:
            return hout

        self.pred = F.softmax(h)
        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(self.pred, t)

        if self.train:
            return self.loss
        else:
            return self.pred

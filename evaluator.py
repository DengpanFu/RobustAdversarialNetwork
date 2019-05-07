#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Dengpan Fu (v-defu@microsoft.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
from torch.nn import functional as F

class Evaluator(object):
    def __init__(self, model, attack, is_cuda=True, verbose=True):
        super(Evaluator, self).__init__()
        self.model = model
        self.attack = attack
        self.is_cuda = is_cuda
        self.verbose = verbose

    def evaluate(self, data_loader, print_freq=1):
        correct, adv_correct, total = 0, 0, 0
        start = time.time()
        for i, data in enumerate(data_loader):
            x, y = data
            if self.is_cuda:
                x = x.cuda()
                y = y.cuda()
            x_adv = self.attack(x, y)
            pred = self.model(x)
            adv_pred = self.model(x_adv)
            pred_y = pred.argmax(1)
            adv_pred_y = adv_pred.argmax(1)
            
            correct += pred_y.eq(y).sum().item()
            adv_correct += adv_pred_y.eq(y).sum().item()
            total += len(y)
            if self.verbose and (i + 1) % print_freq == 0:
                p_str = "[{:3d}|{:3d}] using {:.3f}s ...".format(
                    i + 1, len(data_loader), time.time() - start)
                print(p_str)
        return float(correct)/total, float(adv_correct)/total


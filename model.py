#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Dengpan Fu (v-defu@microsoft.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class Model(nn.Module):
    """ Construct basic model for mnist adversal attack """
    def __init__(self, re_init=False, has_dropout=False):
        super(Model, self).__init__()
        self.re_init = re_init
        self.has_dropout = has_dropout
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(True)
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024, 10)
        if self.has_dropout:
            self.dropout = nn.Dropout()

        if self.re_init:
            self._init_params(self.conv1)
            self._init_params(self.conv2)
            self._init_params(self.fc1)
            self._init_params(self.fc2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)

        if self.has_dropout:
            x = self.dropout(x)

        x = self.fc2(x)

        return x

    def _init_params(self, module, mean=0.1, std=0.1):
        init.normal_(module.weight, std=0.1)
        if hasattr(module, 'bias'):
            init.constant_(module.bias, mean)

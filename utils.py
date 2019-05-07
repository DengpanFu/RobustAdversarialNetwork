#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Dengpan Fu (v-defu@microsoft.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import torch
import torch.nn as nn
from torchvision import datasets, transforms

def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_data_loader(data_dir, batch_size, test_batch_size, num_workers=4):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, 
        pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_model(model, path):
    if not os.path.isfile(path):
        raise IOError('model: {} is non-exists'.format(path))
    if hasattr(model, 'module'):
        module = model.module
    else:
        module = model
    checkpoint = torch.load(path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    module.load_state_dict(state_dict, strict=False)
    print('Params Loaded from: {}'.format(path))

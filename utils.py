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
from torchvision import datasets as D
from torchvision import transforms as T

def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_data_loader(data_name='mnist', data_dir='data/mnist', batch_size=128, 
    test_batch_size=200, num_workers=4):
    if data_name == 'mnist':
        transform_train = T.Compose([T.ToTensor()])
        transform_test = T.Compose([T.ToTensor()])
        train_set = D.MNIST(root=data_dir, train=True, download=True, 
            transform=transform_train)
        test_set = D.MNIST(root=data_dir, train=False, download=True, 
            transform=transform_test)
        img_size, num_class = 28, 10
    elif data_name == 'cifar10':
        transform_train = T.Compose([T.RandomCrop(32, padding=4),
                                     T.RandomHorizontalFlip(),
                                     T.ToTensor(),])
        transform_test = T.Compose([T.ToTensor()])
        train_set = D.CIFAR10(root=data_dir, train=True, download=True, 
            transform=transform_train)
        test_set = D.CIFAR10(root=data_dir, train=False, download=True, 
            transform=transform_test)
        img_size, num_class = 32, 10
    elif data_name == 'stl10':
        transform_train = T.Compose([T.RandomCrop(96, padding=4),
                                     T.RandomHorizontalFlip(),
                                     T.ToTensor(),])
        transform_test = T.Compose([T.ToTensor()])
        train_set = D.STL10(root=data_dir, split='train', download=True, 
            transform=transform_train)
        test_set = D.STL10(root=data_dir, split='test', download=True, 
            transform=transform_test)
        img_size, num_class = 96, 10
    elif data_name == 'imagenet-sub':
        transform_train = D.Compose([
            D.RandomResizedCrop(64, scale=(0.8, 0.9), ratio=(1.0, 1.0)),
            D.RandomHorizontalFlip(),
            D.ToTensor(),])
        transform_test = T.Compose([T.Resize(64), T.ToTensor()])
        train_set = D.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
        test_set = D.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
        img_size, num_class = 64, 143
    else:
        raise ValueError('invalid dataset, current only support {}'.format(
            "mnist, cifar10, stl10, imagenet-sub"))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, 
        shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, num_class, img_size

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

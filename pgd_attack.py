#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Dengpan Fu (v-defu@microsoft.com)

"""
Implementation of attack methods: projected gradient descent (PGD)
    x^(t+1) = Proj_(x+S)(x^(t) + \alpha * sign(grad_x(L(\theta, x^(t), y)))
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
from torch.nn import functional as F


class LinfPGDAttack(object):
    """ Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial point.
    """
    def __init__(self, model, epsilon=0.3, k=40, a=0.01, random_start=True):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.random_start = random_start
        self.x_adv_grad = None

    def record_grad(self, grad_in):
        self.x_adv_grad = grad_in.clone()

    def __call__(self, x, y):
        if self.random_start:
            x_adv = x + x.new(x.size()).uniform_(-self.epsilon, self.epsilon)
        else:
            x_adv = x.clone()

        for i in range(self.k):
            x_adv.requires_grad = True
            pred = F.cross_entropy(self.model(x_adv), y)
            # x_adv.register_hook(self.record_grad)
            pred.backward()
            grad = x_adv.grad.clone()
            # update x_adv
            x_adv = x_adv.detach() + self.a * grad.sign()
            
            # x_adv = np.clip(x_adv, x_adv-self.epsilon, x_adv+self.epsilon)
            indices_up = x_adv > x + self.epsilon
            x_adv[indices_up] = x[indices_up] + self.epsilon
            indices_down = x_adv < x - self.epsilon
            x_adv[indices_down] = x[indices_down] - self.epsilon
            
            x_adv.clamp_(0, 1)

        return x_adv

if __name__ == '__main__':
    from model import Model
    net = Model().cuda()
    attack = LinfPGDAttack(net)
    x = torch.rand(100, 1, 28, 28).cuda()
    y = torch.LongTensor(x.size(0)).random_(10).cuda()
    x_adv = attack(x, y)
    import pdb; pdb.set_trace()  # breakpoint c2b3d99e //


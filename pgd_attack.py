#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Dengpan Fu (v-defu@microsoft.com)

"""
Implementation of attack methods: 
    PGD(Projected Gradient Descent):
        x^(t+1) = Proj_(x+S)(x^(t) + \alpha * sign(grad_x(L(\theta, x^(t), y)))
    MIFGSM(Momentum Iterative Fast Gradient Sign Method):
        g_(t+1) = \mu * g_t + grad_x / || grad_x ||_1
        x^(t+1) = clip(x^(t) + \alpha * sign(g_(t+1)))
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
from torch.nn import functional as F


class LinfPGDAttack(object):
    """ 
        Attack parameter initialization. The attack performs k steps of size 
        alpha, while always staying within epsilon from the initial point.
            IFGSM(Iterative Fast Gradient Sign Method) is essentially 
            PGD(Projected Gradient Descent) 
    """
    def __init__(self, model, epsilon=0.3, k=40, alpha=0.01, random_start=True):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.alpha = alpha
        self.random_start = random_start

    def __call__(self, x, y):
        if self.random_start:
            x_adv = x + x.new(x.size()).uniform_(-self.epsilon, self.epsilon)
        else:
            x_adv = x.clone()
        training = self.model.training
        if training:
            self.model.eval()
        for i in range(self.k):
            x_adv.requires_grad_()
            pred = F.cross_entropy(self.model(x_adv), y)
            pred.backward()
            grad = x_adv.grad.clone()
            # update x_adv
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            
            # x_adv = np.clip(x_adv, x_adv-self.epsilon, x_adv+self.epsilon)
            x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            
            x_adv.clamp_(0, 1)
        if training:
            self.model.train()
        return x_adv

class MIFGSM(object):
    """
        Momentum Iterative Fast Gradient Sign Method(https://arxiv.org/pdf/1710.06081.pdf)
    """
    def __init__(self, model, epsilon=0.02, k=10, mu=1.0, eps=1e-12):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.mu = mu
        self.alpha = self.epsilon / self.k
        self.eps = eps


    def __call__(self, x, y):
        x_adv = x.clone()
        training = self.model.training
        if training:
            self.model.eval()
        grad = None
        for i in range(self.k):
            x_adv.requires_grad_()
            pred = F.cross_entropy(self.model(x_adv), y)
            pred.backward()
            x_grad = x_adv.grad.data
            # norm = torch.mean(torch.abs(x_grad).view((x_grad.shape[0], -1)), dim=1).view((-1, 1, 1, 1))
            norm = x.abs().mean(dim=(1, 2, 3), keepdim=True)
            norm.clamp_(min=self.eps)
            x_grad /= norm
            grad = x_grad if grad is None else self.mu * grad + x_grad
            # update x_adv
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            
            # x_adv = np.clip(x_adv, x_adv-self.epsilon, x_adv+self.epsilon)
            x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            
            x_adv.clamp_(0, 1)
        if training:
            self.model.train()
        return x_adv

__factory = {
    'pgd': LinfPGDAttack, 
    'mifgsm': MIFGSM, 
}
__args_dict = {
    'pgd': ['model', 'epsilon', 'k', 'alpha', 'random_start'], 
    'mifgsm': ['model', 'epsilon', 'k', 'mu'], 
}

def create_attack(attack_method, **kwargs):
    assert(attack_method in __factory), 'invalid attack method'
    kwargs = {k: v for k, v in kwargs.items() if k in __args_dict[attack_method]}
    attack = __factory[attack_method](**kwargs)
    kwargs.pop('model', None)
    print("{:s} attack created with kwargs: {}".format(attack_method, kwargs))
    return attack

if __name__ == '__main__':
    from model import MnistModel as Model
    net = Model().cuda()
    # attack = LinfPGDAttack(net)
    attack = MIFGSM(net)
    x = torch.rand(100, 1, 28, 28).cuda()
    y = torch.LongTensor(x.size(0)).random_(10).cuda()
    x_adv = attack(x, y)
    import pdb; pdb.set_trace()  # breakpoint c2b3d99e //


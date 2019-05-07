#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Dengpan Fu (v-defu@microsoft.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import random
import numpy as np
import pprint
from time import gmtime, strftime
import glob
import torch
import torch.nn as nn
from torch.backends import cudnn

from Logging import Logger
from config import cfg
from model import Model
from evaluator import Evaluator
from pgd_attack import LinfPGDAttack
from utils import *

if __name__ == "__main__":
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
    if not cfg.randomize:
        # set fixed seed
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)

    log_path = os.path.join(cfg.log_dir, cfg.exp_name)
    mkdir_if_missing(log_path)

    if not cfg.no_log:
        log_name = cfg.exp_name + "_eval_log_" + \
                strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + '.txt'
        sys.stdout = Logger(os.path.join(log_path, log_name))

    print("Input Args: ")
    pprint.pprint(cfg)
    _, test_loader = get_data_loader(data_dir=cfg.data_dir, 
        batch_size=cfg.batch_size, test_batch_size=cfg.eval_batch_size, num_workers=4)

    model = Model()

    is_cuda = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()
        is_cuda = True

    if cfg.eval_model is None:
        print('evaluate model is not specified')
        snapshots = []
    elif os.path.isdir(cfg.eval_model):
        snapshots = glob.glob(os.path.join(cfg.eval_model, '*.pth'))
        snapshots = sorted(snapshots, key=lambda x: 
            int(os.path.basename(x).split('_')[1].split('.')[0]))
    elif os.path.isfile(cfg.eval_model):
        snapshots = [cfg.eval_model]
    else:
        snapshots = []

    for snapshot in snapshots:
        load_model(model, snapshot)
        attack = LinfPGDAttack(model=model, epsilon=cfg.epsilon, k=cfg.k, 
                           a=cfg.a, random_start=cfg.random_start)
        evaluator = Evaluator(model=model, attack=attack, is_cuda=is_cuda, verbose=False)
        acc, adv_acc = evaluator.evaluate(test_loader)
        print("natural     accuracy: {:.4f}".format(acc))
        print("adversarial accuracy: {:.4f}".format(adv_acc))

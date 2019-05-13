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

import torch
import torch.nn as nn
from torch.backends import cudnn
from tensorboardX import SummaryWriter

from Logging import Logger
from config import cfg
from model import Model
from trainer import Trainer
from pgd_attack import LinfPGDAttack
from utils import *


if __name__ == '__main__':
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
    snap_path = os.path.join(cfg.snap_dir, cfg.exp_name)
    mkdir_if_missing(snap_path)

    summary_writer = None
    if not cfg.no_log:
        log_name = cfg.exp_name + "_log_" + \
                strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + '.txt'
        sys.stdout = Logger(os.path.join(log_path, log_name))
        summary_writer = SummaryWriter(log_dir=log_path)

    print("Input Args: ")
    pprint.pprint(cfg)
    train_loader, test_loader, num_class, img_size = get_data_loader(
        data_name=cfg.data_name, data_dir=cfg.data_dir, batch_size=cfg.batch_size, 
        test_batch_size=cfg.eval_batch_size, num_workers=4)

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, 
                        weight_decay=0.0005, amsgrad=False)
    is_cuda = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()
        is_cuda = True

    attack = LinfPGDAttack(model=model, epsilon=cfg.epsilon, k=cfg.k, 
                           a=cfg.alpha, random_start=cfg.random_start)
    trainer = Trainer(model=model, attack=attack, optimizer=optimizer, 
                      summary_writer=summary_writer, is_cuda=True, 
                      output_freq=cfg.output_freq, print_freq=cfg.print_freq)

    trainer.reset()
    for epoch in range(cfg.max_epoch):
        trainer.train(epoch, train_loader)
        if cfg.save_freq < 1:
            save_current = False
        else:
            save_current = (epoch + 1) % cfg.save_freq == 0 \
                or epoch == 0 or epoch == cfg.max_epoch - 1
        if save_current:
            if hasattr(model, 'module'):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            dict_to_save = {'state_dict': state_dict, 
                            'epoch': epoch + 1}
            fpath = os.path.join(snap_path, 'checkpoint_' + 
                        str(epoch + 1) + '.pth')
            torch.save(dict_to_save, fpath)

    trainer.close()
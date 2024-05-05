#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from datasets import load_dataset
from noise2noise import Noise2Noise
from argparse import ArgumentParser
from utils import get_max_min


def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('-t', '--train-dir', help='training set path', default='./../data/train')
    parser.add_argument('-v', '--valid-dir', help='test set path', default='./../data/valid')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./../ckpts')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--report-interval', help='batch report interval', default=250, type=int)
    parser.add_argument('-ts', '--train-size', help='size of train dataset', type=int)
    parser.add_argument('-vs', '--valid-size', help='size of valid dataset', type=int)

    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters (b1, b2, eps, weight_decay)', nargs='+', default=[0.9, 0.99, 1e-8, 0], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=4, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=100, type=int)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2', 'hdr'], default='l1', type=str)
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true')

    # Corruption parameters
    parser.add_argument('-n', '--noise-type', help='noise type',
        choices=['gaussian', 'poisson', 'text', 'mc'], default='gaussian', type=str)
    parser.add_argument('-p', '--noise-param', help='noise parameter (e.g. std for gaussian)', default=50, type=float)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='random crop size', default=128, type=int)
    parser.add_argument('--clean-targets', help='use clean targets for training', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    """Trains Noise2Noise."""

    # Parse training parameters
    params = parse_args()

    params.train_dir = "data/train_ll"
    params.train_size = 10
    params.valid_dir = "data/valid_ll"
    params.valid_size = 3
    params.ckpt_save_path = "../ckpts"
    params.nb_epochs = 60
    params.batch_size = 3
    params.loss = "l2"
    params.noise_type = "gaussian"
    params.noise_param = 0
    params.crop_size = 128
    params.plot_stats = True
    params.cuda = True
    params.seed = 42
    params.report_interval = 1
    params.learning_rate = 1e-4
    params.adam = [0.9, 0.99, 1e-8, 0]

    #mxx = 1.669760584831238

    #max_val_train, min_val_train = get_max_min(params.train_dir, "mat")

    # Train/valid datasets
    train_loader = load_dataset(
        params.train_dir,
        params.train_size,
        params,
        shuffled=True,
        is_train=True
        )

    valid_loader = load_dataset(
        params.valid_dir,
        params.valid_size,
        params,
        shuffled=False,
        is_train=False
        )

    # Initialize model and train
    n2n = Noise2Noise(params, trainable=True)
    n2n.train(train_loader, valid_loader)

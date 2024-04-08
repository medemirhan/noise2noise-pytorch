#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from datasets import load_dataset
from noise2noise import Noise2Noise

from argparse import ArgumentParser


def parse_args():
    """Command-line argument parser for testing."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('-d', '--data', help='dataset root path', default='../data')
    parser.add_argument('--load-ckpt', help='load model checkpoint')
    parser.add_argument('--show-output', help='pop up window to display outputs', default=0, type=int)
    parser.add_argument('--cuda', help='use cuda', action='store_true')

    # Corruption parameters
    parser.add_argument('-n', '--noise-type', help='noise type',
        choices=['gaussian', 'poisson', 'text', 'mc'], default='gaussian', type=str)
    parser.add_argument('-v', '--noise-param', help='noise parameter (e.g. sigma for gaussian)', default=50, type=float)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='image crop size', default=256, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    """Tests Noise2Noise."""

    # Parse test parameters
    params = parse_args()

    params.data = "data/test"
    params.load_ckpt = "../ckpts/gaussian-20240407_175022/n2n-epoch10-1487.72949.pt"
    params.noise_type = "gaussian"
    params.noise_param = 50
    params.crop_size = 256
    params.show_output = 1
    params.cuda = True
    params.seed = 42

    #mxx = 1.669760584831238

    # Initialize model and test
    n2n = Noise2Noise(params, trainable=False)
    params.redux = False
    params.clean_targets = True
    test_loader = load_dataset(
        params.data,
        0,
        params,
        shuffled=False,
        single=True,
        is_train=False,
        normalization_path="data/norm.pkl"
        )
    n2n.load_model(params.load_ckpt)
    n2n.test(test_loader, show=params.show_output)

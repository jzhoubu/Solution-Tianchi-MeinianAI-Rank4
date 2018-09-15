# -*- coding:utf-8 -*-

import sys

import os
import logging

from conf import root_dir

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--version', type=str, default='v2', help='Model version')
args = parser.parse_args()

data_path = os.path.join(root_dir, 'data/dataset/train_data_part1_full.pickle')
data_numeric_path = '/data1/liangzheng/mndata/dataset/0429-Data2-debug4.pkl'
pretrain_path = os.path.join(root_dir, 'data/pretrain/word2vec.vectors.64d.nb')

def train():
    version = args.version
    if version == 'v2':
        from model.model_v2_double_cnn.train import train as train2
        logging.critical("Using model version 2, DoubleMultiCnn")
        train2(data_path, pretrain_path)
    elif version == "v1":
        from model.model_v1_cnn.train import train as train1
        logging.critical("Using model version 1, MultiCnn")
        train1(data_path, pretrain_path)
    elif version == "v3":
        from model.model_v3_double_cnn_join_numeric.train import train as train3
        logging.critical("Using model version 1, DoubleMultiCnnJoinNumeric")
        train3(data_path, data_numeric_path, pretrain_path)
    else:
        raise ValueError("Wrong model version {}".format(version))

if __name__ == "__main__":
    train()

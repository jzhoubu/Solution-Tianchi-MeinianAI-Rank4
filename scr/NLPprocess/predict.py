# -*- coding:utf-8 -*-
import os
import argparse
import logging
logging.getLogger().setLevel(logging.INFO)

from conf import root_dir

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--version', type=str, default='v2', help='Model version')
parser.add_argument('--trained_dir', type=str, help='path of trained result')
parser.add_argument('-o', '--output', type=str, default='features')
args = parser.parse_args()

data_path = os.path.join(root_dir, 'data/dataset/train_data_part1_test.pickle')

def write_features():
    trained_dir = args.trained_dir
    # trained_dir = './trained_result/trained_results_1525230888/'
    if args.version == 'v2':
        from model.model_v2_double_cnn.predict import write_features as write2
        logging.critical("Using model version 2, DoubleMultiCnn")
        write2(data_path, trained_dir)
    elif args.version == "v1":
        from model.model_v1_cnn.train import write_features as write1
        logging.critical("Using model version 1, MultiCnn")
        write1(data_path, trained_dir)
    else:
        raise ValueError("Wrong model version {}".format(args.version))

def write_result():
    trained_dir = args.trained_dir
    # trained_dir = './trained_result/trained_results_1525252754/'
    if args.version == 'v2':
        from model.model_v2_double_cnn.predict import write_result as write2
        logging.critical("Using model version 2, DoubleMultiCnn")
        write2(data_path, trained_dir, test=True)
    elif args.version == "v1":
        from model.model_v1_cnn.predict import write_result as write1
        logging.critical("Using model version 1, MultiCnn")
        write1(data_path, trained_dir, test=True)
    else:
        raise ValueError("Wrong model version {}".format(args.version))

if __name__ == "__main__":
    if args.output == 'result':
        write_result()
    else:
        write_features()

# -*- coding:utf-8 -*-
from conf import preprocess_config
from data_helper.format import clean_str_column
from data_helper.preprocess.process_enum import DealWithEnum


def process(data, config=preprocess_config, is_dict=False):
    features = []
    str_process = config['str_process'] if is_dict else config.str_process

    # 排序, 确保顺序相同
    str_process = sorted(str_process)

    for column in str_process:
        data[column] = data[column].apply(lambda x:clean_str_column(x))
        features.append(column)
    return data, features

def process_v2(data, config=preprocess_config, is_dict=False):
    features = []
    short_features = []
    str_process = config['str_process'] if is_dict else config.str_process
    short_str_process = config['short_str_process'] if is_dict else config.short_str_process

    # 排序, 确保顺序相同
    str_process = sorted(str_process)
    short_str_process = sorted(short_str_process)

    for column in str_process:
        data[column] = data[column].apply(lambda x:clean_str_column(x))
        features.append(column)
    for column in short_str_process:
        data[column] = data[column].apply(lambda x:clean_str_column(x))
        short_features.append(column)
    return data, features, short_features

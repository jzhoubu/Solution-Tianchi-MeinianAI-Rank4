
import re
import os
import yaml
import math

from data_helper.value import *

def clean_data_column(data):
    """
        清洗数据, 对字符串与数值混合的数据进行处理, 对Nan进行填补, 并去掉Nan数量太多的行
    :param data:
    :return:
    """
    data_without_nan = data.dropna(axis=0, how='any')
    data_without_nan = list(set(data_without_nan))
    data_without_nan_type = list(map(lambda x:type(x), data_without_nan))

    float_type_count = 0
    str_type_count = 0

    for word, word_type in zip(data_without_nan, data_without_nan_type):
        real_type = get_real_type(word, word_type)
        if real_type == float:
            float_type_count += 1
        elif real_type == str:
            str_type_count += 1

    column_str_res = []
    column_float_res = []

    # 如果该列中没有有效数据, 跳过
    if float_type_count + str_type_count == 0:
        pass

    # 如果浮点型数据非常少, 可将浮点型全部置为BLANK
    elif float_type_count / (float_type_count + str_type_count) < 0.05:
        data = data.fillna(BLANK_WORD)
        for word in data:
            column_str_res.append(clean_str_column(word))

    # 如果字符串型数据非常少, 可将浮点型全部置为BLANK
    elif str_type_count / (float_type_count + str_type_count) < 0.05:
        data = data.fillna(BLANK_FLOAT)
        for word in data:
            column_float_res.append(clean_float_column(word))

    # 如果字符串数据与浮点型数据数量相当, 分成两列数据
    else:
        for word in data:
            if type(word) == float or type(word) == int:
                column_str_res.append(BLANK_WORD)
                column_float_res.append(clean_float_column(word))
            elif type(word) == str:
                column_str_res.append(clean_str_column(word))
                column_float_res.append(BLANK_FLOAT)
            else:
                column_str_res.append(BLANK_WORD)
                column_float_res.append(BLANK_FLOAT)

    return column_str_res, column_float_res

def get_real_type(word, word_type):
    """
        返回数据的真实类型
    :param word:
    :param word_type:
    :return:
    """
    if not isinstance(word, (float, int, str)):
        return None
    elif isinstance(word, str) and not is_decimal(word):
        return str
    else:
        return float

    
def clean_str_column(word, default=BLANK_WORD):
    if isinstance(word, str):
        if not is_decimal(word):
            return word

    return default

def clean_float_column(word, default=BLANK_FLOAT):
    """
        返回清洗过后的浮点型数据
    :param word:
    :param default:
    :return:
    """
    if isinstance(word, (float, int)):
        if not math.isnan(word):
            return float(word)
    return default


def is_decimal(word):
    return word.encode('UTF-8').isdigit() or bool(re.match('^\d+\.\d+$', word))

def is_alpha_num(s):
    return s.encode('UTF-8').isalnum()

def is_alpha(s):
    return s.encode('UTF-8').isalpha()

dbc_format_dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dbc.yml")
dbc_format_dict = yaml.load(open(dbc_format_dict_path, 'r', encoding='utf-8'))
# print(dbc_format_dict)

def to_dbc_and_lower_case(word):
    """
        全角转半角, 大写转小写
    :param word:
    :return:
    """
    word = to_dbc_case(word)
    word = to_lower_case(word)

    return word

def to_dbc_case(word):
    """
        全角转半角
    :param word:
    :return:
    """
    new_word = str()
    for w in word:
        new_word += dbc_format_dict.get(w, w)
    return new_word

def to_lower_case(word):
    """
        大写转小写
    :param word:
    :return:
    """
    word = word.lower()
    return word

def to_upper_case(word):
    """
        小写转大写
    :param word:
    :return:
    """
    word = word.upper()
    return word
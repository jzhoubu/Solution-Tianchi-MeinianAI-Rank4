import logging
logging.getLogger().setLevel(logging.INFO)

import re
import os

from conf import root_dir
from data_helper.dictionary import load_vocab_file_to_list
from data_helper.value import *
from data_helper.format import to_dbc_and_lower_case, is_decimal

import jieba

user_dict_path = os.path.join(root_dir, 'data/dictionary/dict.dic')
jieba.load_userdict(user_dict_path)

def sentence_segment(sentence, word_seg=False):
    res = []

    if not sentence == BLANK_WORD:
        sentence = sentence.strip()
        sentence = to_dbc_and_lower_case(sentence)
        sentence_list = sentence_seg_by_comma(sentence)
    else:
        sentence_list = [ BLANK_WORD ]

    if word_seg:
        for sentence in sentence_list:
            word_seg = word_segment(sentence)
            if len(word_seg) > 0:
                res.append(word_seg)
    else:
        res = sentence_list
    return res

def word_segment(sentence):
    if not sentence == BLANK_WORD:
        word_seg = list(jieba.cut(sentence))
        word_seg = clean_symbol(word_seg)
        word_seg = clean_stopword(word_seg)
        word_seg = clean_numeric(word_seg)
        return word_seg
    else:
        return [ BLANK_WORD ]

def convert_numeric(word):
    if is_decimal(word):
        num = float(word)
        multiple = int(num // 5)
        if multiple < 0:
            return "<NUM/NEG>"
        elif multiple > 15:
            return "<NUM/BIG>"
        elif 0 < multiple < 0.3:
            return "<NUM/1/3>"
        elif 0.3 <= multiple < 0.7:
            return "<NUM/2/3>"
        elif 0.7 <= multiple < 1.0:
            return "<NUM/3/3>"
        else:
            return "<NUM/" + str(multiple) + ">"
    else:
        return word

def clean_numeric(word_seg):
    res = []
    for word in word_seg:
        word = convert_numeric(word)
        res.append(word)
    return res

def clean_symbol(word_seg):
    res = []
    for word in word_seg:
        if word.__contains__("x") or word.__contains__("X"):
            res.extend(separate_numeric_area(word))
        elif not word in [",", "。", "，", "?", "!", "\"", "(", ")", ":", " "]:
            res.append(word)
    return res

stopwords_dict_path = os.path.join(root_dir, './data/dictionary/stopwords.dic')
stopwords = load_vocab_file_to_list(stopwords_dict_path)

def separate_numeric_area(word):
    words = re.split("x|X", word)
    if len(words) == 1:
        return [ word ]
    if min(map(lambda w: float(is_decimal(w)) if len(w) > 0 else 1.0, words)) > 0:
        words_sep = (' x '.join(words)).strip().split(' ')
        return words_sep
    else:
        return [ word ]

def clean_stopword(word_seg):
    res = []
    for word in word_seg:
        if not word in stopwords:
            res.append(word)
    return res

def sentence_seg_by_comma(sentence):
    """
        Segment sentence by comma "," and full stop "。"
    :param sentence:
    :return:
    """
    res = []

    if isinstance(sentence, list):
        for sent in sentence:
            res.extend(sentence_seg_by_comma(sent))
    else:
        res.extend(re.split('。', sentence))
        # res.extend(re.split(',|。|，', sentence))

    while "" in res:
        res.remove("")

    return res

def get_max_counts_and_length(sentence_data):
    """
        计算分词后数据的最大句子数量与最大句子长度(词数量)
    :param sentence_data: [[[xx, xx, xx], [xx, xx]], [[xx], [xxx,xx]]]
    :return:
    """
    max_count, max_length = 0, 0
    for sentence_list in sentence_data:
        count = len(sentence_list)
        max_count = max(count, max_count)
        for sentence in sentence_list:
            length = len(sentence)
            max_length = max(length, max_length)
    return max_count, max_length

if __name__ == '__main__':
    print(separate_numeric_area('x7'))
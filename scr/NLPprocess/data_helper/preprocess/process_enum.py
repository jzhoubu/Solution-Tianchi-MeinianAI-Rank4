# -*- coding:utf-8 -*-

from data_helper.preprocess.process_regex import *
from data_helper.preprocess.utils import contains_vocab

def deal_with_normal(func):
    def decorate(sentence):
        if is_normal(sentence):
            return float(False)
        else:
            return func(sentence)
    return decorate

def is_normal(sentence):
    if not isinstance(sentence, str):
        return True
    return contains_vocab(sentence, description_normal) or sentence in description_void

class DealWithEnum:

    @staticmethod
    def deal_with_enum(sentence, column):
        """
            通过反射调用不同列的预处理函数
        :param sentence:
        :param column:
        :return:
        """
        function_name = "deal_with_" + column

        if hasattr(DealWithEnum, function_name):
            return getattr(DealWithEnum, "deal_with_" + column)(sentence)

        else:
            return float(not is_normal(sentence))

    @staticmethod
    @deal_with_normal
    def deal_with_0124(sentence):
        return float(bool(not contains_vocab(sentence, description_0124)))

    @staticmethod
    @deal_with_normal
    def deal_with_0206(sentence):
        return float(bool(not contains_vocab(sentence, description_0206)))

    @staticmethod
    @deal_with_normal
    def deal_with_0212(sentence):
        return float(bool(not contains_vocab(sentence, description_0212)))

    @staticmethod
    @deal_with_normal
    def deal_with_0421(sentence):
        return float(bool(contains_vocab(sentence, description_0421)))
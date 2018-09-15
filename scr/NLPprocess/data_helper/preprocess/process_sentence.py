# -*- coding:utf-8 -*-
from data_helper.format import to_dbc_and_lower_case
from data_helper.preprocess.process_regex import *
from data_helper.preprocess.utils import endswith_vocab
from data_helper.value import BLANK_WORD

def deal_with_normal(func):
    def decorate(sentence):
        sentence = sentence.strip()
        sentence = to_dbc_and_lower_case(sentence)
        if sentence in description_void:
            return BLANK_WORD
        else:
            return func(sentence)
    return decorate

class DealWithSentence:

    @staticmethod
    def deal_with_sentence(sentence, column):
        """
            通过反射调用不同列的预处理函数
        :param sentence:
        :param column:
        :return:
        """
        function_name = "deal_with_" + column

        if hasattr(DealWithSentence, function_name):
            return getattr(DealWithSentence, "deal_with_" + column)(sentence)

        else:
            return sentence

    @staticmethod
    @deal_with_normal
    def deal_with_0102(sentence):
        sentence = split_by_symbol(sentence, symbol=":", keyword=keyword_0102)
        sentence = split_by_symbol(sentence, symbol="、", keyword=keyword_0102, first_stop=True)
        return sentence

    @staticmethod
    @deal_with_normal
    def deal_with_0222(sentence):
        sentence = split_by_symbol(sentence, symbol=":", keyword=keyword_0222)
        return sentence

    @staticmethod
    @deal_with_normal
    def deal_with_0409(sentence):
        sentence = split_by_symbol(sentence, symbol=":", keyword=keyword_0409)
        return sentence

    @staticmethod
    @deal_with_normal
    def deal_with_0539(sentence):
        sentence = split_by_symbol(sentence, symbol=":", keyword=keyword_0539)
        return sentence

    @staticmethod
    @deal_with_normal
    def deal_with_0709(sentence):
        sentence = split_by_symbol(sentence, symbol=":", keyword=keyword_0709)
        return sentence

    @staticmethod
    @deal_with_normal
    def deal_with_0978(sentence):
        sentence_split = sentence.split(":")
        new_sentence = ""
        for i in range(len(sentence_split)):
            sent = sentence_split[i]
            if not sent.__contains__(","):
                new_sentence += sent
            else:
                sent_split = sent.split(",")
                new_sentence += ",".join(sent_split[:-1]) + "。"
                new_sentence += sent_split[-1]
            new_sentence += ":"
        new_sentence = new_sentence.strip(":")
        return new_sentence

    @staticmethod
    @deal_with_normal
    def deal_with_0539(sentence):
        sentence = split_by_symbol(sentence, symbol=":", keyword=keyword_0539)
        return sentence

def split_by_symbol(sentence, symbol, keyword, first_stop=False):
    if first_stop:
        index_of_symbol = sentence.find(symbol)
        sentence_split = [sentence[:index_of_symbol], sentence[(index_of_symbol+1):]]
    else:
        sentence_split = sentence.split(symbol)

    new_sentence = ""
    for i in range(len(sentence_split)):
        sent = sentence_split[i]
        end_word = endswith_vocab(sent, keyword)
        if end_word and not end_word == sent:
            new_sentence += sent[:-len(end_word)] + "。"
            new_sentence += end_word
        else:
            new_sentence += sent
        new_sentence += symbol

    new_sentence = new_sentence.strip(symbol)
    return new_sentence

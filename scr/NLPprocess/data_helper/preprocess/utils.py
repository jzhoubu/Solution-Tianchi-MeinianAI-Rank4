# -*- coding:utf-8 -*-

def endswith_vocab(word, vocab):
    """
        If word endswith one of members in vocab, return this member, else, return None
        For efficiency, ensure the vocab has been sorted by length inversely
    :param word:
    :param vocab:
    :return:
    """
    if not isinstance(word, str):
        return None

    for w in vocab:
        if word.endswith(w):
            return w
    return None

def contains_vocab(word, vocab):
    """
        If word contains one of members in vocab, return this member, else, return None
    :param word:
    :param vocab:
    :return:
    """
    if not isinstance(word, str):
        return None

    for w in vocab:
        if word.__contains__(w):
            return w
    return None
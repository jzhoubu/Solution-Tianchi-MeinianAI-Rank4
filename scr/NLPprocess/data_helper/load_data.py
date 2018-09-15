import itertools
import logging
import pickle
import re
import time
from collections import Counter

import numpy as np

from conf import preprocess_config
from data_helper.format import clean_data_column, is_alpha_num, is_alpha, is_decimal
from data_helper.preprocess import processor
from data_helper.preprocess.process_sentence import DealWithSentence
from data_helper.segment import sentence_segment, get_max_counts_and_length, word_segment
from data_helper.value import *
from data_helper.trie_tree import Trie

logging.getLogger().setLevel(logging.INFO)

result = preprocess_config.result
str_process = preprocess_config.str_process


def load_data(data_path, word_embedding_size=128, max_count=3, max_length=15, shuffle=False, pretrain_embedding_path=None):
    """
        数据加载函数
    :param data_path:
    :param word_embedding_size:
    :param max_count:
    :param max_length:
    :return:
    """
    data = pickle.load(open(data_path, 'rb'))
    # drop掉vid
    data.drop(["vid"], axis=1, inplace=True)

    data, features = processor.process(data)

    if shuffle:
        data = data.reindex(np.random.permutation(data.index))

    # columns_str_features_dict: [sentence_count * sequence_length]
    columns_str_features_dict, columns_sequences_count_dict, sequences_count, sequence_length = \
        process_str_features(data[features], max_count=max_count, max_length=max_length)

    logging.critical("Train data sequences count is {}".format(sequences_count))
    logging.critical("Train data sequence length is {}".format(sequence_length))

    merged_str_features_list = merge_str_features(columns_str_features_dict, sequences_count, sequence_length)

    # 生成词典
    word_embedding, vocabulary, vocabulary_inv, vocab_tree= load_embedding(sentences=merged_str_features_list,
                                    word_embedding_size=word_embedding_size,
                                    pretrain_embedding_path=pretrain_embedding_path)

    input_x, input_x_ratio = build_str_features_data(merged_str_features_list, vocabulary, vocab_tree)
    input_x = np.array(input_x, dtype=np.int32)
    input_x_ratio = np.array(input_x_ratio, dtype=np.float32)

    input_y_data = data[result].fillna(0.0)
    input_y = np.array(input_y_data)
    input_y = np.array(input_y, dtype=np.float32)

    return input_x, input_y, input_x_ratio, word_embedding, vocabulary, vocabulary_inv, \
           sequences_count, sequence_length, columns_sequences_count_dict

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    # shuffle表示重新排列
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        # 迭代返回batch中的数据
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index: end_index]

def add_word(vocabulary, vocabulary_inv, vocab_tree, word):
    if not vocabulary.__contains__(word):
        vocabulary[word] = len(vocabulary_inv)
        vocabulary_inv.append(word)
        vocab_tree.insert(word)

def load_embedding(sentences, word_embedding_size=128, pretrain_embedding_path=None):
    """
        生成词向量
    :param vocabulary:
    :param embedding_dim:
    :return:
    """

    word_embedding = {}
    vocabulary = {}
    vocabulary_inv = []
    vocab_tree = Trie()
    # 先将padding_word放到词典第一位
    for word in PADDING_WORDS:
        add_word(vocabulary, vocabulary_inv, vocab_tree, word)

    if pretrain_embedding_path:
        import gensim as gs
        model = gs.models.KeyedVectors.load_word2vec_format(pretrain_embedding_path, binary=False)
        for word in PADDING_WORDS:
            if word in model.wv.vocab:
                word_embedding[word] = model[word]

        for word in model.wv.vocab:
            add_word(vocabulary, vocabulary_inv, vocab_tree, word)
            word_embedding[word] = model[word]

        logging.critical(
            'The pretrain embedding load complete, the size of vocabulary is {} {}'.format(
                len(vocabulary_inv), len(vocabulary)))

    build_vocab(sentences, vocabulary, vocabulary_inv, vocab_tree)
    bound = np.sqrt(6.0) / np.sqrt(len(vocabulary))
    for word in vocabulary:
        if not word_embedding.__contains__(word):
            word_embedding[word] = np.random.uniform(-bound, bound, word_embedding_size)

    return word_embedding, vocabulary, vocabulary_inv, vocab_tree

def build_vocab(sentences, vocabulary, vocabulary_inv, vocab_tree, ratio=1.0):
    """
        生成词典
    :param sentences:
    :param ratio:
    :return:
    """
    word_counts = Counter(itertools.chain(*list(itertools.chain(*sentences))))

    vocab_size =int(len(word_counts) * ratio)

    for word, _ in word_counts.most_common(vocab_size):
        if not vocab_tree.search(word):
            # 处理中文词
            if not is_alpha_num(word):
                # 词长
                N = len(word)
                last_index = 0
                for k in range(N):
                    # 从上一个切词的index处开始检索子词
                    if k < last_index:
                        continue
                    index = -1
                    i = k
                    frag = word[k:i + 1]
                    while i < N:
                        _ = vocab_tree.starts_with_or_is(frag)
                        if _ < 0:
                            break
                        # 如果是单词, 记录检索到子词的index
                        elif _ == 0:
                            index = i
                        i += 1
                        frag = word[k:i + 1]
                    # 如果index小于0， 表示未找到词, 直接将目前的字作为新词
                    if index < 0:
                        new_word = word[k:k + 1]
                        add_word(vocabulary, vocabulary_inv, vocab_tree, new_word)
                    last_index = index + 1 if index > 0 else k + 1
            # 处理英文词
            elif is_alpha(word):
                add_word(vocabulary, vocabulary_inv, vocab_tree, word)

def get_word_id(word, vocabulary, vocab_tree, subword_length=5):
    """
        单词在词典中的index
    :param word:
    :param vocabulary:
    :return:
    """
    word_array = []
    word_ratio = []

    if vocab_tree.search(word):
        word_array.append(vocabulary[word])
    elif re.search('[\u4e00-\u9fa5]+', word) == None:
        word_array.append(vocabulary[BLANK_WORD])
    else:
        N = len(word)
        last_index = 0
        for k in range(N):
            if word_array.__len__() >= subword_length:
                break
            if k < last_index:
                continue
            index = -1
            i = k
            frag = word[k:i + 1]
            while i < N:
                _ = vocab_tree.starts_with_or_is(frag)
                if _ < 0:
                    break
                elif _ == 0:
                    index = i
                i += 1
                frag = word[k:i + 1]
            if index >= 0:
                word_array.append(vocabulary[word[k:index + 1]])
            else:
                word_array.append(vocabulary[PAD_WORD])
            last_index = index + 1 if index > 0 else k + 1

    real_len = word_array.__len__()
    for i in range(real_len):
        word_ratio.append(1 / real_len)

    while word_array.__len__() < subword_length:
        word_array.append(vocabulary[PAD_WORD])
        word_ratio.append(0.0)

    return word_array, word_ratio

def merge_str_features(columns_list, data_size, columns_str_features_dict, sequences_count, sequence_length):
    merged_str_features_list = []
    for i in range(data_size):
        merged_str_features = []
        for col in columns_list:
            merged_str_features.extend(columns_str_features_dict[col][i])
        # 长度与句子数量检验
        # [sequences_count * sequence_length]
        assert len(merged_str_features) == sequences_count
        assert len(merged_str_features[0]) == sequence_length
        # 加入到列表中
        # [data_size * sequences_count * sequence_length]
        merged_str_features_list.append(merged_str_features)
    return merged_str_features_list

def build_str_features_data(str_features_list, vocabulary, vocab_tree):
    """
        将字符串数据生成训练用的浮点型矩阵data
    :param str_features_list: 合并后的句子数据
    :param vocabulary:
    :return:
    """
    x = []
    x_ratio = []
    for str_features in str_features_list:
        str_feature_id = []
        str_features_ratio = []
        for sentence in str_features:
            sentence_id = []
            sentence_ratio = []
            for word in sentence:
                word_id, word_ratio = get_word_id(word, vocabulary, vocab_tree)
                sentence_id.append(word_id)
                sentence_ratio.append(word_ratio)
            str_feature_id.append(sentence_id)
            str_features_ratio.append(sentence_ratio)
        x.append(str_feature_id)
        x_ratio.append(str_features_ratio)
    return x, x_ratio

def process_str_features(data, max_count=3, max_length=20):
    """
        对字符型数据进行预处理
    :param data:
    :param max_count: 一个列单元的最大句子数量
    :param max_length: 最大句子长度(词数量)
    :return:
    """
    columns_sequences_count_dict = {}
    columns_sequence_length_dict = {}
    columns_str_features_dict = {}

    stt = time.time()

    for col in data.columns:
        sentence_data = []
        for sentence in data[col]:
            sentence = DealWithSentence.deal_with_sentence(sentence, col)
            assert type(sentence) == str
            sentence_data.append(sentence_segment(sentence, word_seg=True))
        columns_str_features_dict[col] = sentence_data
        col_max_sequences_count, col_max_sequence_length = get_max_counts_and_length(sentence_data)
        columns_sequences_count_dict[col] = col_max_sequences_count
        columns_sequence_length_dict[col] = col_max_sequence_length

    edt = time.time()
    logging.info("Sentences segment complete, runing time: {} seconds".format((edt-stt)))

    max_sequences_count = min(max(columns_sequences_count_dict.values()), max_count)
    max_sequence_length = min(max(columns_sequence_length_dict.values()), max_length)

    # 根据最大的句子数, 更新列-句子数dict
    for col, value in columns_sequences_count_dict.items():
        columns_sequences_count_dict[col] = min(value, max_sequences_count)

    logging.critical("Max sequences count: {}".format(max_sequences_count))
    logging.critical("Max sequence length: {}".format(max_sequence_length))

    total_sequences_count = 0
    for col, sentence_data in columns_str_features_dict.items():
        sequences_count = columns_sequences_count_dict[col]
        total_sequences_count += sequences_count
        padded_sentence_data = []
        for sentence_list in sentence_data:
            padded_sentence_list = pad_sentence_list(sentence_list, sequences_count, max_sequence_length)
            padded_sentence_data.append(padded_sentence_list)
        columns_str_features_dict[col] = padded_sentence_data

    return columns_str_features_dict, columns_sequences_count_dict, total_sequences_count, max_sequence_length

def process_predict_str_features(data, columns_sequences_count_dict, max_sequence_length):
    """
        根据之前训练保存的列--句子数量对，对data生成features数据
    :param data:
    :param columns_sequences_count_dict:
    :param max_sequence_length:
    :return:
    """
    columns_str_features_dict = {}
    for col in data.columns:
        sentence_data = []
        for sentence in data[col]:
            sentence = DealWithSentence.deal_with_sentence(sentence, col)
            assert type(sentence) == str
            sentence_data.append(sentence_segment(sentence, word_seg=True))
        columns_str_features_dict[col] = sentence_data
    for col, sentence_data in columns_str_features_dict.items():
        sequences_count = columns_sequences_count_dict[col]
        padded_sentence_data = []
        for sentence_list in sentence_data:
            padded_sentence_list = pad_sentence_list(sentence_list, sequences_count, max_sequence_length)
            padded_sentence_data.append(padded_sentence_list)
        columns_str_features_dict[col] = padded_sentence_data
    return columns_str_features_dict

def pad_sentence_list(sentence_list, max_count, max_length):
    """
        将data里的句子长度与句子数量补至输入维度的值
    :param sentence_list:
    :param max_count:
    :param max_length:
    :return:
    """
    res = []
    for index in range(max_count):
        if index >= len(sentence_list):
            res.append([PAD_WORD] * max_length)
        else:
            sentence = sentence_list[index]
            num_padding = max_length - len(sentence)
            if num_padding < 0:
                padded_sentence = sentence[0:max_length]
            else:
                padded_sentence = sentence + [ PAD_WORD ] * num_padding
            res.append(padded_sentence)
    return res

def load_text(data_path, output_path):
    with open(data_path, 'rb') as data_pickle:
        data = pickle.load(data_pickle)
        # drop掉vid
        data.drop(["vid"], axis=1, inplace=True)

    output = []
    features_columns = list(set(data.columns) - set(result))
    for col in features_columns:
        str_res, float_res = clean_data_column(data[col])
        output.extend(str_res)

    with open(output_path, 'w', encoding='utf-8') as output_file:
        for text in output:
            text = text.strip()
            if text == BLANK_WORD:
                continue
            if text:
                words = word_segment(text)
                output_file.write(" ".join(words))
                output_file.write("\n")

if __name__ == "__main__":
    data_path = "../data/dataset/train_data_part1.pickle"
    # load_data(data_path)
    output_path = "../data/text_data.txt"
    load_text(data_path, output_path)

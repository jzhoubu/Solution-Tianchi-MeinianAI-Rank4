# -*- coding:utf-8 -*-

from data_helper.load_data import *
from data_helper.preprocess import processor

logging.getLogger().setLevel(logging.INFO)

result = preprocess_config.result
str_process = preprocess_config.str_process

def load_data(data_path, word_embedding_size=128, max_count1=3, max_length1=15, max_count2=3, max_length2=6, shuffle=False, pretrain_embedding_path=None):
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

    # data, features, short_features_columns = processor.process_v2(data)
    data, features_columns, short_features_columns = processor.process_v2(data)

    if shuffle:
        data = data.reindex(np.random.permutation(data.index))

    # columns_str_features_dict: [sentence_count * sequence_length]
    columns_str_features_dict1, columns_sequences_count_dict1, sequences_count1, sequence_length1 = \
        process_str_features(data[features_columns], max_count=max_count1, max_length=max_length1)

    logging.critical("Train data long text sequences count is {}".format(sequences_count1))
    logging.critical("Train data long text sequence length is {}".format(sequence_length1))

    merged_str_features_list1 = merge_str_features(features_columns, len(data), columns_str_features_dict1, sequences_count1, sequence_length1)

    columns_str_features_dict2, columns_sequences_count_dict2, sequences_count2, sequence_length2 = \
        process_str_features(data[short_features_columns], max_count=max_count2, max_length=max_length2)

    logging.critical("Train data short text sequences count is {}".format(sequences_count2))
    logging.critical("Train data short text sequence length is {}".format(sequence_length2))

    merged_str_features_list2 = merge_str_features(short_features_columns, len(data), columns_str_features_dict2, sequences_count2, sequence_length2)

    # 生成词典
    word_embedding, vocabulary, vocabulary_inv, vocab_tree = load_embedding(sentences=merged_str_features_list1 + merged_str_features_list2,
                                                                word_embedding_size=word_embedding_size,
                                                                pretrain_embedding_path=pretrain_embedding_path)

    input_x1, input_x1_ratio = build_str_features_data(merged_str_features_list1, vocabulary, vocab_tree)
    input_x1 = np.array(input_x1, dtype=np.int32)
    input_x1_ratio = np.array(input_x1_ratio, dtype=np.float32)

    input_x2, input_x2_ratio = build_str_features_data(merged_str_features_list2, vocabulary, vocab_tree)
    input_x2 = np.array(input_x2, dtype=np.int32)
    input_x2_ratio = np.array(input_x2_ratio, dtype=np.float32)

    input_y_data = data[result].fillna(0.0)
    input_y = np.array(input_y_data)
    input_y = np.array(input_y, dtype=np.float32)

    return input_x1, input_x2, input_x1_ratio, input_x2_ratio, input_y, word_embedding, vocabulary, vocabulary_inv, \
           sequences_count1, sequences_count2, sequence_length1, sequence_length2, \
           columns_sequences_count_dict1, columns_sequences_count_dict2
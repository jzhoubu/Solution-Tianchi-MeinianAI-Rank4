# -*- coding:utf-8 -*-
import json
import logging
import pickle
import os

import numpy as np
import shutil
import tensorflow as tf
import pandas as pd

from conf import root_dir

logging.getLogger().setLevel(logging.INFO)

from data_helper import load_data
from data_helper.preprocess import processor
from model.model_v1_cnn.multi_cnn_model import MultiCnnModel


def load_trained_params(trained_dir):
    params = json.loads(open(trained_dir + 'trained_parameters.json').read())
    vocabulary = json.loads(open(trained_dir + 'vocabulary.json').read())

    with open(trained_dir + 'word_embedding.pickle', 'rb') as input_file:
        word_embedding = pickle.load(input_file)
    word_embedding_mat = np.array(word_embedding, dtype=np.float32)

    with open(trained_dir + 'preprocess.json', 'r') as input_file:
        preprocess_config = json.load(input_file)

    with open(trained_dir + 'columns_sequences_count.json', 'r') as input_file:
        columns_sequences_count_dict = json.load(input_file)

    return params, preprocess_config, columns_sequences_count_dict, vocabulary, word_embedding_mat

def predict(data, trained_dir, output='features'):
    params, preprocess_config, columns_sequences_count_dict, vocabulary, word_embedding_mat = \
        load_trained_params(trained_dir)

    sequences_count = params['sequences_count']
    sequence_length = params['sequence_length']

    data, features_columns = processor.process(data, config=preprocess_config, is_dict=True)
    columns_str_features_dict = load_data.process_predict_str_features(data[features_columns], columns_sequences_count_dict, sequence_length)

    merged_str_features_list = load_data.merge_str_features(features_columns, len(data), columns_str_features_dict, sequences_count, sequence_length)

    input_x = load_data.build_str_features_data(merged_str_features_list, vocabulary)

    model = MultiCnnModel(
        batch_size=params['batch_size'],
        sequences_count=sequences_count,
        sequence_length=sequence_length,
        word_embedding_size=params['word_embedding_size'],
        word_embedding_mat=word_embedding_mat,
        vocabulary_size=len(vocabulary),
        conv_filter_sizes=list(map(int, params['conv_filter_sizes'].split(','))),
        num_filters=params['num_filters'],
        hidden_size=params['hidden_size'],
        output_size=len(preprocess_config['result']),
        dropout_keep_prob=1.0,
        is_training=False,
        decay_steps=params['decay_steps'],
        learning_rate=params['learning_rate'],
        clip_gradients=params['clip_gradients'],
        l2_lambda=params['l2_lambda'],
    )

    checkpoint_file = trained_dir + 'best_model.ckpt'
    saver = tf.train.Saver(tf.global_variables())
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(model.sess, checkpoint_file)
    logging.critical('{} has been loaded'.format(checkpoint_file))

    total_size = np.ceil(len(input_x) / params['batch_size'])

    batches = load_data.batch_iter(input_x, params['batch_size'], num_epochs=1, shuffle=False)
    batch_count = 0
    output_data = []

    if output == 'result':
        predict_function = model.predict_step
    else:
        predict_function = model.predict_output_features_step

    for batch in batches:
        x_batch = batch
        output_data_batch = predict_function(x_batch)
        output_data.append(output_data_batch)
        batch_count += 1
        if batch_count % 20 == 0:
            progress = ('%.2f' % (batch_count / total_size * 100))
            logging.info('Progress at : {}% data'.format(progress))
    output_data = np.vstack(output_data)

    return output_data, params, preprocess_config, features_columns, columns_sequences_count_dict

def write_features(data_path, trained_dir):
    data = pickle.load(open(data_path, 'rb'))
    
    if not trained_dir.endswith('/'):
        trained_dir += '/'

    timestamp = trained_dir.split('/')[-2].split('_')[-1]
    output_dir = os.path.join(root_dir, 'predicted_results_' + timestamp)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    output_features, params, _, features_columns, columns_sequences_count_dict = predict(data, trained_dir)

    features_data = pd.DataFrame()
    features_data['vid'] = data['vid']
    output_col = 0
    for col in features_columns:
        sequences_count = columns_sequences_count_dict.get(col, 0)
        for i in range(sequences_count * params['hidden_size']):
            features_data["dim" + str(i) + "_" + col] = output_features[:, output_col]
            output_col += 1

    logging.critical("Actual Output features dimension is {}".format(output_col))
    if not output_col == output_features.shape[1]:
        logging.critical("输出的features维度与模型理论输出不符")

    logging.critical("Predict output features finish, saved in {}".format(output_dir))

    with open(os.path.join(output_dir, "features_data.csv"), 'w', encoding='utf-8') as output_file:
        features_data.to_csv(output_file, index=False)

    with open(os.path.join(output_dir, "features_data.pkl"), 'wb', encoding='utf-8') as output_file:
        pickle.dump(features_data, output_file, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_dir, "features_columns.txt"), 'w', encoding='utf-8') as output_file:
        for col in features_columns:
            output_file.write(col)
            output_file.write("\n")

def write_result(data_path, trained_dir, test=False):
    data = pickle.load(open(data_path, 'rb'))

    if not trained_dir.endswith('/'):
        trained_dir += '/'

    timestamp = trained_dir.split('/')[-2].split('_')[-1]
    output_dir = os.path.join(root_dir, 'predicted_result_' + timestamp)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    output_result, _, preprocess_config, _, _ = predict(data, trained_dir, output='result')
    logging.critical("Predict output result finish, saved in {}".format(output_dir))

    result_data = pd.DataFrame()
    result_data['vid'] = data['vid']
    for i, col in enumerate(preprocess_config['result']):
        result_data[col] = output_result[:, i]

    if test:
        total_loss = 0.0
        count = 0
        for col in preprocess_config['result']:
            for i in range(len(result_data[col])):
                per_loss = loss(data[col][i], result_data[col][i])
                total_loss += per_loss
                count += 1
        logging.critical("Testing loss is {}".format(total_loss / count))

    with open(os.path.join(output_dir, "result_data.csv"), 'w', encoding='utf-8') as output_file:
        result_data.to_csv(output_file, index=False)

    with open(os.path.join(output_dir, "result_data.pkl"), 'wb') as output_file:
        pickle.dump(result_data, output_file, pickle.HIGHEST_PROTOCOL)

def loss(input, output):
    return np.square(np.log(input + 1.0) - np.log(output + 1.0))
    
    



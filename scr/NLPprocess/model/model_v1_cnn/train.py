import json
import logging
import os
import pickle
import shutil
import time

import numpy as np
import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split

from conf import preprocess_yaml, root_dir
from data_helper.load_data import load_data, batch_iter
from model.model_v1_cnn.multi_cnn_model import MultiCnnModel

CHECKPOINT_PATH = "./checkpoints/"
SUMMARY_PATH = "./summary/"

logging.getLogger().setLevel(logging.INFO)

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.003, 'learning_rate')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size for training/evaluating.')
tf.app.flags.DEFINE_integer('decay_steps', 1000, 'how many steps before decay learning_rate.')
tf.app.flags.DEFINE_float('decay_rate', 0.9, 'Rate of decay for learning rate.')
tf.app.flags.DEFINE_boolean('is_training', True, 'is_training.true: training, false:testing')
tf.app.flags.DEFINE_integer('num_epochs', 80, 'num of epochs to run.')
tf.app.flags.DEFINE_integer('validate_step', 100, 'how many step to validate.')

tf.app.flags.DEFINE_integer('sequences_count', 4, 'maximum of the number of sequences in one column')
tf.app.flags.DEFINE_integer('sequence_length', 18, 'sequence length')
tf.app.flags.DEFINE_integer('word_embedding_size', 64, 'dimension of word embedding')
tf.app.flags.DEFINE_string('conv_filter_sizes', '2,3', 'convolution filter size')
tf.app.flags.DEFINE_integer('num_filters', 32, 'output channels of each filters')

tf.app.flags.DEFINE_integer('hidden_size', 8, 'hidden size of output dense layer')

tf.app.flags.DEFINE_float('dropout_keep_prob', 0.6, 'dropout')
tf.app.flags.DEFINE_float('l2_lambda', 0.0005, 'l2_regularization')
tf.app.flags.DEFINE_float('clip_gradients', 3.0, 'the normalized gradients')

tf.app.flags.DEFINE_integer('max_step', 9999999, 'the most step of training')

data_path = os.path.join(root_dir, 'data/dataset/train_data_part1_simple.pickle')

def train(data_path, pretrain_path=None):
    # 生成训练数据
    data_x, data_y, word_embedding, vocabulary, vocabulary_inv,\
        sequences_count, sequence_length, columns_sequences_count_dict\
        = load_data(data_path,
                    word_embedding_size=FLAGS.word_embedding_size,
                    max_count=FLAGS.sequences_count,
                    max_length=FLAGS.sequence_length,
                    shuffle=True,
                    pretrain_embedding_path=pretrain_path)

    # 词向量矩阵
    word_embedding_mat = [word_embedding[word] for index, word in enumerate(vocabulary_inv)]
    word_embedding_mat = np.array(word_embedding_mat, dtype=np.float32)

    # split the original dataset into train set and test set
    data_x, data_x_test, data_y, data_y_test = train_test_split(
        data_x, data_y, test_size=0.1)

    # split the train set into train set and dev set
    data_x_train, data_x_dev, data_y_train, data_y_dev = train_test_split(
        data_x, data_y, test_size=0.1)

    logging.info('data_train: {}, data_dev: {}, data_test: {}'.format(len(data_x_train), len(data_x_dev), len(data_x_test)))

    timestamp = str(int(time.time()))
    trained_dir = './trained_result/trained_results_' + timestamp + '/'
    if os.path.exists(trained_dir):
        shutil.rmtree(trained_dir)
    os.makedirs(trained_dir)
    logging.critical('The trained result is saved in {}'.format(trained_dir))

    params = dict()
    params['learning_rate'] = FLAGS.learning_rate
    params['batch_size'] = FLAGS.batch_size
    params['decay_steps'] = FLAGS.decay_steps
    params['decay_rate'] = FLAGS.decay_rate
    params['num_epochs'] = FLAGS.num_epochs

    params['max_count'] = FLAGS.sequences_count
    params['sequences_count'] = sequences_count
    params['sequence_length'] = sequence_length
    params['word_embedding_size'] = FLAGS.word_embedding_size
    params['conv_filter_sizes'] = FLAGS.conv_filter_sizes
    params['num_filters'] = FLAGS.num_filters
    params['hidden_size'] = FLAGS.hidden_size

    params['dropout_keep_prob'] = FLAGS.dropout_keep_prob
    params['l2_lambda'] = FLAGS.l2_lambda
    params['clip_gradients'] = FLAGS.clip_gradients

    with open(trained_dir + 'trained_parameters.json', 'w') as outfile:
        json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)

    with open(trained_dir + 'preprocess.json', 'w') as outfile:
        json.dump(preprocess_yaml, outfile, indent=4, sort_keys=True, ensure_ascii=False)

    with open(trained_dir + 'columns_sequences_count.json', 'w') as outfile:
        json.dump(columns_sequences_count_dict, outfile, indent=4, ensure_ascii=False)

    # 初始化Model
    model = MultiCnnModel(
        # batch_size
        batch_size=FLAGS.batch_size,
        # sequence数量
        sequences_count=sequences_count,
        # sequence长度
        sequence_length=sequence_length,
        # 词向量维度
        word_embedding_size=FLAGS.word_embedding_size,
        # 词向量矩阵
        word_embedding_mat=word_embedding_mat,
        # 词典size
        vocabulary_size=len(vocabulary),
        # 卷积层filter_size
        conv_filter_sizes=list(map(int, FLAGS.conv_filter_sizes.split(','))),
        # 卷积层输出通道数
        num_filters=FLAGS.num_filters,
        # 输出dense层维度
        hidden_size=FLAGS.hidden_size,
        # 输出维度
        output_size=data_y.shape[1],
        # dropout
        dropout_keep_prob=FLAGS.dropout_keep_prob,
        # 是否训练
        is_training=FLAGS.is_training,
        # 衰减步数
        decay_steps=FLAGS.decay_steps,
        # 衰减速率
        decay_rate=FLAGS.decay_rate,
        # 学习速率
        learning_rate=FLAGS.learning_rate,
        # 梯度裁剪
        clip_gradients=FLAGS.clip_gradients,
        # 正二范数权值
        l2_lambda=FLAGS.l2_lambda,
    )

    # summary_dir = SUMMARY_PATH + 'summary_' + timestamp + '/'
    # if os.path.exists(summary_dir):
    #     shutil.rmtree(summary_dir)
    # os.makedirs(summary_dir)
    # logging.critical('The summary information is saved in {}'.format(summary_dir))
    # train_writer = tf.summary.FileWriter(summary_dir + 'train', model.sess.graph)

    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    checkpoint_dir = CHECKPOINT_PATH + 'check_points_' + timestamp + '/'
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

    init = tf.global_variables_initializer()
    model.sess.run(init)

    saver = tf.train.Saver(tf.global_variables())

    # Train start here
    total_size = np.ceil(len(data_x_train) / FLAGS.batch_size) * FLAGS.num_epochs

    best_loss = 999
    best_step = 0
    batch_count = 0

    train_batches = batch_iter(list(zip(data_x_train, data_y_train)), FLAGS.batch_size, FLAGS.num_epochs)

    for train_batch in train_batches:
        batch_count += 1
        x_train_batch, y_train_batch = zip(*train_batch)
        train_loss = model.train_step(x_train_batch, y_train_batch)
        current_step = tf.train.global_step(model.sess, model.global_step)

        #Evaluate the model
        if current_step % FLAGS.validate_step == 0:
            progress = ('%.2f' % (batch_count / total_size * 100))
            dev_batches = batch_iter(list(zip(data_x_dev, data_y_dev)), FLAGS.batch_size, 1)
            dev_loss, dev_batch_num = 0.0, 0
            for dev_batch in dev_batches:
                dev_batch_num += 1
                x_dev_batch, y_dev_batch = zip(*dev_batch)
                dev_loss_batch = model.dev_step(x_dev_batch, y_dev_batch)
                dev_loss += dev_loss_batch

            dev_loss /= dev_batch_num
            logging.info('Progress at : {}% examples, Train Loss is {}, Dev Loss is {}'.format(progress, train_loss, dev_loss))

            if dev_loss < best_loss:
                best_loss, best_step = dev_loss, current_step
                path = saver.save(model.sess, checkpoint_prefix, global_step=current_step)
                logging.critical(
                    'Best loss {} at step {}'.format(best_loss, best_step))

        # if current step exceeds max step, stop training
        if current_step >= FLAGS.max_step:
            break

    logging.critical('Training is complete, testing the best model on x_test and y_test')
    
    # close summary writer
    # train_writer.close()

    # evaluate x_test and y_test
    saver.restore(model.sess, checkpoint_prefix + '-' + str(best_step))
    
    #Save the model files to trained_dir
    saver.save(model.sess, trained_dir + 'best_model.ckpt')

    test_batches = batch_iter(list(zip(data_x_test, data_y_test)), FLAGS.batch_size, 1)
    test_loss, test_batch_num = 0.0, 0
    for test_batch in test_batches:
        test_batch_num += 1
        x_test_batch, y_test_batch = zip(*test_batch)
        test_loss_batch = model.dev_step(x_test_batch, y_test_batch)
        test_loss += test_loss_batch

    test_loss /= test_batch_num
    logging.critical('Loss on test set is {}'.format(test_loss))

    with open(trained_dir + 'vocabulary.json', 'w') as outfile:
        json.dump(vocabulary, outfile, indent=4, ensure_ascii=False)

    word_embedding_mat = model.sess.run(model.word_embedding_mat)
    with open(trained_dir + 'word_embedding.pickle', 'wb') as outfile:
        pickle.dump(word_embedding_mat, outfile, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    train()

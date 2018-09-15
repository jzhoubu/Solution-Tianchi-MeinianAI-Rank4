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

from conf import preprocess_yaml, root_dir, preprocess_config
from data_helper.load_data_v2 import load_data
from data_helper.load_data import batch_iter
from model.model_v2_double_cnn.double_multi_cnn_model import DoubleMultiCnnModel
from model.model_v2_double_cnn.predict import predict_by_model

CHECKPOINT_PATH = "./checkpoints/"
SUMMARY_PATH = "./summary/"

logging.getLogger().setLevel(logging.INFO)

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.0025, 'learning_rate')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size for training/evaluating.')
tf.app.flags.DEFINE_integer('decay_steps', 1200, 'how many steps before decay learning_rate.')
tf.app.flags.DEFINE_float('decay_rate', 0.85, 'Rate of decay for learning rate.')
tf.app.flags.DEFINE_boolean('is_training', True, 'is_training.true: training, false:testing')
tf.app.flags.DEFINE_integer('num_epochs', 180, 'num of epochs to run.')
tf.app.flags.DEFINE_integer('validate_step', 100, 'how many step to validate.')

tf.app.flags.DEFINE_integer('sequences_count1', 4, 'maximum of the number of sequences in one column')
tf.app.flags.DEFINE_integer('sequences_count2', 3, 'maximum of the number of sequences in one column')

tf.app.flags.DEFINE_integer('sequence_length1', 18, 'sequence length')
tf.app.flags.DEFINE_integer('sequence_length2', 8, 'sequence length')

tf.app.flags.DEFINE_integer('word_embedding_size', 64, 'dimension of word embedding')

tf.app.flags.DEFINE_string('conv_filter_sizes1', '2,3,4', 'convolution filter size')
tf.app.flags.DEFINE_string('conv_filter_sizes2', '1,2,3', 'convolution filter size')
tf.app.flags.DEFINE_integer('num_filters1', 64, 'output channels of each filters')
tf.app.flags.DEFINE_integer('num_filters2', 32, 'output channels of each filters')

tf.app.flags.DEFINE_integer('hidden_size1', 16, 'hidden size of output dense layer')
tf.app.flags.DEFINE_integer('hidden_size2', 16, 'hidden size of output dense layer')

tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'dropout')
tf.app.flags.DEFINE_float('l2_lambda', 0.00004, 'l2_regularization')
tf.app.flags.DEFINE_float('clip_gradients', 3.0, 'the normalized gradients')

tf.app.flags.DEFINE_integer('subword_length', 5, 'subword_length')

tf.app.flags.DEFINE_float('learning_rate_threshold', 0.0004, 'slow down the decay of lr if lr lower than threshold')
tf.app.flags.DEFINE_integer('decay_steps_scale', 2, 'magnification of decay steps to slow down the decay of lr')

tf.app.flags.DEFINE_integer('max_step', 9999999, 'the most step of training')

def train(data_path, pretrain_embedding_path=None, pretrain_model_path=None):
    # 生成训练数据
    data_x_1, data_x_2, data_x_1_ratio, data_x_2_ratio, data_y, word_embedding, vocabulary, vocabulary_inv,\
        sequences_count1, sequences_count2, sequence_length1, sequence_length2, \
    columns_sequences_count_dict1, columns_sequences_count_dict2 \
        = load_data(data_path,
                    word_embedding_size=FLAGS.word_embedding_size,
                    max_count1=FLAGS.sequences_count1,
                    max_length1=FLAGS.sequence_length1,
                    max_count2=FLAGS.sequences_count2,
                    max_length2=FLAGS.sequence_length2,
                    shuffle=True,
                    pretrain_embedding_path=pretrain_embedding_path)

    # 词向量矩阵
    word_embedding_mat = [word_embedding[word] for index, word in enumerate(vocabulary_inv)]
    word_embedding_mat = np.array(word_embedding_mat, dtype=np.float32)

    # split the original dataset into train set and test set
    data_x_1, data_x_1_test, data_x_2, data_x_2_test, \
    data_x_1_ratio, data_x_1_ratio_test, data_x_2_ratio, data_x_2_ratio_test, \
    data_y, data_y_test = train_test_split(
        data_x_1, data_x_2, data_x_1_ratio, data_x_2_ratio, data_y, test_size=0.1)

    # split the train set into train set and dev set
    data_x_1_train, data_x_1_dev, data_x_2_train, data_x_2_dev, \
    data_x_1_ratio_train, data_x_1_ratio_dev, data_x_2_ratio_train, data_x_2_ratio_dev, \
    data_y_train, data_y_dev = train_test_split(
        data_x_1, data_x_2, data_x_1_ratio, data_x_2_ratio, data_y, test_size=0.1)

    logging.info('data_train: {}, data_dev: {}, data_test: {}'.format(len(data_x_1_train), len(data_x_1_dev), len(data_x_1_test)))

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

    params['max_count1'] = FLAGS.sequences_count1
    params['max_count2'] = FLAGS.sequences_count2

    params['sequences_count1'] = sequences_count1
    params['sequences_count2'] = sequences_count2
    params['sequence_length1'] = sequence_length1
    params['sequence_length2'] = sequence_length2
    params['word_embedding_size'] = FLAGS.word_embedding_size

    params['conv_filter_sizes1'] = FLAGS.conv_filter_sizes1
    params['conv_filter_sizes2'] = FLAGS.conv_filter_sizes2
    params['num_filters1'] = FLAGS.num_filters1
    params['num_filters2'] = FLAGS.num_filters2

    params['hidden_size1'] = FLAGS.hidden_size1
    params['hidden_size2'] = FLAGS.hidden_size2

    params['dropout_keep_prob'] = FLAGS.dropout_keep_prob
    params['l2_lambda'] = FLAGS.l2_lambda
    params['clip_gradients'] = FLAGS.clip_gradients

    params['subword_length'] = FLAGS.subword_length

    with open(trained_dir + 'trained_parameters.json', 'w') as outfile:
        json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)

    with open(trained_dir + 'preprocess.json', 'w') as outfile:
        json.dump(preprocess_yaml, outfile, indent=4, sort_keys=True, ensure_ascii=False)

    with open(trained_dir + 'columns_sequences_count1.json', 'w') as outfile:
        json.dump(columns_sequences_count_dict1, outfile, indent=4, ensure_ascii=False)

    with open(trained_dir + 'columns_sequences_count2.json', 'w') as outfile:
        json.dump(columns_sequences_count_dict2, outfile, indent=4, ensure_ascii=False)

    conv_filter_sizes1 = list(map(int, FLAGS.conv_filter_sizes1.split(',')))
    conv_filter_sizes2 = list(map(int, FLAGS.conv_filter_sizes2.split(',')))

    # 初始化Model
    model = DoubleMultiCnnModel(
        # batch_size
        batch_size=FLAGS.batch_size,
        # sequence数量
        sequences_count=(sequences_count1, sequences_count2),
        # sequence长度
        sequence_length=(sequence_length1, sequence_length2),
        # 词向量维度
        word_embedding_size=FLAGS.word_embedding_size,
        # 词向量矩阵
        word_embedding_mat=word_embedding_mat,
        # 词典size
        vocabulary_size=len(vocabulary),
        # 卷积层filter_size
        conv_filter_sizes=(conv_filter_sizes1, conv_filter_sizes2),
        # 卷积层输出通道数
        num_filters=(FLAGS.num_filters1, FLAGS.num_filters2),
        # 输出dense层维度
        hidden_size=(FLAGS.hidden_size1, FLAGS.hidden_size2),
        # 输出维度
        output_size=data_y.shape[1],
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
        # 是否在训练
        is_training=True,
        #subword length
        subword_length=FLAGS.subword_length
    )
    saver = tf.train.Saver(tf.global_variables())

    if pretrain_model_path:
        saver.restore(model.sess, tf.train.latest_checkpoint(pretrain_model_path))

    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    checkpoint_dir = CHECKPOINT_PATH + 'check_points_' + timestamp + '/'
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

    # Train start here
    total_size = np.ceil(len(data_x_1_train) / FLAGS.batch_size) * FLAGS.num_epochs
    epoch_size = np.ceil(total_size / FLAGS.num_epochs)
    batch_size_larger = 0

    best_loss = 999
    best_step = 0
    batch_count = 0

    train_batches = batch_iter(list(zip(data_x_1_train, data_x_2_train, data_x_1_ratio_train, data_x_2_ratio_train, data_y_train)), FLAGS.batch_size, FLAGS.num_epochs)

    while True:
        try:
            train_batch = train_batches.__next__()
        except StopIteration:
            break

        batch_count += 1 * int(model.batch_size / FLAGS.batch_size)
        x_1_train_batch, x_2_train_batch, x_1_ratio_train_batch, x_2_ratio_train_batch, y_train_batch = zip(*train_batch)
        train_loss = model.train_step(x_1_train_batch, x_2_train_batch, x_1_ratio_train_batch, x_2_ratio_train_batch, y_train_batch, FLAGS.dropout_keep_prob)
        current_step = tf.train.global_step(model.sess, model.global_step)

        #Evaluate the model
        if current_step % FLAGS.validate_step == 0:
            progress = ('%.2f' % (batch_count / total_size * 100))
            dev_batches = batch_iter(list(zip(data_x_1_dev, data_x_2_dev, data_x_1_ratio_dev, data_x_2_ratio_dev, data_y_dev)), FLAGS.batch_size, 1)
            dev_loss, dev_batch_num = 0.0, 0
            for dev_batch in dev_batches:
                dev_batch_num += 1
                x_1_dev_batch, x_2_dev_batch, x_1_ratio_dev_batch, x_2_ratio_dev_batch, y_dev_batch = zip(*dev_batch)
                dev_loss_batch = model.dev_step(x_1_dev_batch, x_2_dev_batch, x_1_ratio_dev_batch, x_2_ratio_dev_batch, y_dev_batch)
                dev_loss += dev_loss_batch

            dev_loss /= dev_batch_num
            logging.info('Progress at : {}% examples, Train Loss is {}, Dev Loss is {}'.format(progress, train_loss, dev_loss))

            if dev_loss < best_loss:
                best_loss, best_step = dev_loss, current_step
                path = saver.save(model.sess, checkpoint_prefix, global_step=current_step)
                logging.critical(
                    'Best loss {} at step {}'.format(best_loss, best_step))

        if batch_count % epoch_size == 0:

            logging.critical('Learning rate is {}'.format(model.sess.run(model.learning_rate)))

            changed = True

            if batch_size_larger == 0 and best_loss < 0.0405:
                pass

            elif batch_size_larger == 1 and best_loss < 0.0395:
                pass

            elif batch_size_larger == 2 and best_loss < 0.0385:
                pass
            else:
                changed = False

            if changed:
                model.batch_size = FLAGS.batch_size * np.power(2, batch_size_larger + 1)
                new_epochs = max(int(FLAGS.num_epochs * (1-batch_count / total_size)), int(FLAGS.num_epochs / 4))
                train_batches = batch_iter(list(zip(data_x_1_train, data_x_2_train, data_x_1_ratio_train, data_x_2_ratio_train, data_y_train)), model.batch_size,
                                            new_epochs)
                batch_size_larger += 1
                logging.critical("enlarge batch_size to {}".format(model.batch_size))

            # change learning_rate in training
            scale = int(FLAGS.learning_rate_threshold / model.sess.run(model.learning_rate))
            if scale >= 1:
                model.decay_steps = FLAGS.decay_steps * FLAGS.decay_steps_scale * scale
            logging.critical("Learning rate decay steps changed to {}".format(model.decay_steps))

            # if current step exceeds max step, stop training
            if current_step >= FLAGS.max_step:
                break
    logging.critical('Training is complete, testing the best model on x_test and y_test')

    # evaluate x_test and y_test
    saver.restore(model.sess, checkpoint_prefix + '-' + str(best_step))

    #Save the model files to trained_dir
    saver.save(model.sess, trained_dir + 'best_model.ckpt')

    test_batches = batch_iter(list(zip(data_x_1_test, data_x_2_test, data_x_1_ratio_test, data_x_2_ratio_test, data_y_test)), FLAGS.batch_size, 1)
    test_loss, test_batch_num = 0.0, 0
    for test_batch in test_batches:
        test_batch_num += 1
        x_1_test_batch, x_2_test_batch, x_1_ratio_test_batch, x_2_ratio_test_batch, y_test_batch = zip(*test_batch)
        test_loss_batch = model.dev_step(x_1_test_batch, x_2_test_batch, x_1_ratio_test_batch, x_2_ratio_test_batch, y_test_batch)
        test_loss += test_loss_batch

    test_loss /= test_batch_num
    logging.critical('Loss on test set is {}'.format(test_loss))

    predict_by_model(preprocess_config, columns_sequences_count_dict1, columns_sequences_count_dict2, vocabulary, model)

    with open(trained_dir + 'vocabulary.json', 'w') as outfile:
        json.dump(vocabulary, outfile, indent=4, ensure_ascii=False)

    with open(trained_dir + 'word_embedding.pickle', 'wb') as outfile:
        pickle.dump(word_embedding_mat, outfile, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    train()

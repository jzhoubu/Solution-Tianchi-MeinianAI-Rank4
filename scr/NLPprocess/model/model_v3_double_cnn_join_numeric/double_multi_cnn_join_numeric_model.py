import tensorflow as tf
import numpy as np
import logging
from model.utils import leaky_relu, batch_norm_layer

logging.getLogger().setLevel(logging.INFO)

class DoubleMultiCnnJoinNumericModel():
    def __init__(self, batch_size, sequences_count, sequence_length, word_embedding_size, word_embedding_mat,
                 vocabulary_size, conv_filter_sizes, num_filters, output_size, numeric_size, hidden_size, subword_length, is_training,
                 l2_lambda=0.0001, learning_rate=0.001, decay_steps=1000, decay_rate=0.9, clip_gradients=3.0):

        # 句子数量
        self.sequences_count1, self.sequences_count2 = sequences_count
        # 句子长度
        self.sequence_length1, self.sequence_length2 = sequence_length
        # 卷积层维度
        self.conv_filter_sizes1, self.conv_filter_sizes2, = conv_filter_sizes
        # 卷积层通道数
        self.num_filters1, self.num_filters2 = num_filters
        # hidden_size
        self.hidden_size1, self.hidden_size2 = hidden_size

        # 每个batch的大小
        self.batch_size = batch_size
        # 词向量维度
        self.word_embedding_size = word_embedding_size
        # 词向量矩阵
        self.word_embedding_mat = word_embedding_mat
        # 词典size
        self.vocabulary_size = vocabulary_size
        # 输出维度
        self.output_size = output_size
        # 数值维度
        self.numeric_size = numeric_size
        # dropout
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # True: 训练,  False: 预测
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        # l2正则项系数
        self.l2_lambda = l2_lambda
        # 学习率
        self.learning_rate = learning_rate
        # 递减步数
        self.decay_steps = decay_steps
        # 递减速率
        self.decay_rate = decay_rate
        # 梯度裁剪
        self.clip_gradients = clip_gradients

        # 子词长度
        self.subword_length = subword_length

        # 输入1 [batch_size * sequences_count * sequence_length]
        self.input_x_1 = tf.placeholder(tf.int32, [None, self.sequences_count1, self.sequence_length1, self.subword_length])
        # 输入2
        self.input_x_2 = tf.placeholder(tf.int32, [None, self.sequences_count2, self.sequence_length2, self.subword_length])
        # 输入1子词比例
        self.input_x_1_ratio = tf.placeholder(tf.float32, [None, self.sequences_count1, self.sequence_length1, self.subword_length])
        # 输入2子词比例
        self.input_x_2_ratio = tf.placeholder(tf.float32, [None, self.sequences_count2, self.sequence_length2, self.subword_length])

        # 输出数值型矩阵
        self.input_x_3 = tf.placeholder(tf.float32, [None, self.numeric_size])

        # 输出
        self.input_y = tf.placeholder(tf.float32, [None, self.output_size])
        # global_step
        self.global_step = tf.Variable(0, trainable=False, name='Global_step')

        self.instantiate_word_embedding()

        # cnn层
        # self.multi_conv_features = self.multi_conv_layer()
        self.multi_conv_features1 = self.multi_conv_layer(self.input_x_1,
                                                          self.input_x_1_ratio,
                                                          sequences_count=self.sequences_count1,
                                                          sequence_length=self.sequence_length1,
                                                          num_filters=self.num_filters1,
                                                          conv_filter_sizes=self.conv_filter_sizes1,
                                                          hidden_size=self.hidden_size1,
                                                          scope='multi_conv_1')
        self.multi_conv_features2 = self.multi_conv_layer(self.input_x_2,
                                                          self.input_x_2_ratio,
                                                          sequences_count=self.sequences_count2,
                                                          sequence_length=self.sequence_length2,
                                                          num_filters=self.num_filters2,
                                                          conv_filter_sizes=self.conv_filter_sizes2,
                                                          hidden_size=self.hidden_size2,
                                                          scope='multi_conv_2')
        # 输出inference层
        self.inference()

        if is_training:
            # loss function计算
            self.loss = self.compute_loss()
            # train
            self.train_op = self.train()

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def conv_layer(self, sequence, sequence_length, num_filters, conv_filter_sizes, hidden_size, scope='conv_layer', scope_num='0'):
        """
            cnn层, 每列的数据共用同样的cnn层
        :param sequence:
        :param scope:
        :return:
        """
        pooled_outputs = []
        for i, filter_size in enumerate(conv_filter_sizes):
            with tf.variable_scope(scope + '-%s' % filter_size, reuse=tf.AUTO_REUSE):
                conv_filter = tf.get_variable('filter-%s' % filter_size,
                                              [filter_size, self.word_embedding_size, 1, num_filters])
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, conv_filter)
                # [batch_size * sequence_length * word_embedding_size * 1]
                conv = tf.nn.conv2d(sequence, conv_filter, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                # [batch_size * (sequence_length - filter_size + 1) * 1 * num_filters]
                conv = batch_norm_layer(conv, is_training=self.is_training)
                bias = tf.get_variable('conv_bias-%s' % filter_size, [num_filters])
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, bias)
                h = tf.tanh(tf.nn.bias_add(conv, bias))
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name='pool')

                # [batch_size * 1 * 1 * num_filters]
                pooled_outputs.append(pooled)

        # [batch_size * 1 * 1 * num_filters_total]
        self.h_pool = tf.concat(pooled_outputs, -1)
        # [batch_size * num_filters_total]
        num_filters_total = len(conv_filter_sizes) * num_filters
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope('dropout'):
            conv_feature = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)

        with tf.variable_scope('{}_{}_full_connect'.format(scope, scope_num), reuse=tf.AUTO_REUSE):
            w = tf.get_variable('projection_w', shape=[num_filters_total, hidden_size],
                                initializer=tf.contrib.layers.xavier_initializer())
            g = tf.get_variable("g", initializer=tf.sqrt(1.0 / hidden_size))
            b = tf.get_variable("projection_b", shape=[hidden_size], initializer=tf.constant_initializer(np.sqrt(1/hidden_size)))
            normed_v = g * w * tf.rsqrt(tf.reduce_sum(tf.square(w)))
            conv_feature = tf.matmul(conv_feature, normed_v) + b

        return conv_feature

    def multi_conv_layer(self, input_x, input_x_ratio, sequences_count, sequence_length, num_filters, conv_filter_sizes, hidden_size, scope='multi_conv_layer'):
        # [batch_size * sequences_count * sequence_length * word_embedding_size]
        embedded_input_x = self.embedding_lookup_subword(self.word_embedding_mat, input_x, input_x_ratio)
        # [batch_size * sequences_count * word_embedding_size]
        # flatted_input_x = tf.reduce_mean(embedded_input_x, axis=2, keep_dims=False)
        # [batch_size * sequences_count * 1]
        # aligned_weight = self.get_align_weights(flatted_input_x, scope=scope)

        with tf.variable_scope(scope):
            conv_features = []
            sub_sequences = tf.split(embedded_input_x, sequences_count, axis=1)
            for i, sub_sequence in enumerate(sub_sequences):
                sub_sequence = tf.squeeze(sub_sequence, axis=1)
                sub_sequence = tf.expand_dims(sub_sequence, axis=-1)
                sub_conv_feature = self.conv_layer(sub_sequence,
                                                   sequence_length=sequence_length,
                                                   num_filters=num_filters,
                                                   conv_filter_sizes=conv_filter_sizes,
                                                   hidden_size=hidden_size,
                                                   scope_num=str(i),
                                                   )
                conv_features.append(sub_conv_feature)

            # [batch_size * num_filters_total] * sequence_counts --> [batch_size * sequences_count * num_filters_total]
            multi_conv_features = tf.stack(conv_features, axis=1)

            # [batch_size * sequences_count * num_filters_total]
            #multi_conv_features = tf.multiply(multi_conv_features, aligned_weight)

            with tf.name_scope('dropout'):
                multi_conv_features = tf.nn.dropout(multi_conv_features, keep_prob=self.dropout_keep_prob)

            return multi_conv_features

    def inference(self):
        
        _, sequences_count1, features_size1 = self.multi_conv_features1.get_shape().as_list()
        _, sequences_count2, features_size2 = self.multi_conv_features2.get_shape().as_list()
        
        self.output_features1 = tf.reshape(self.multi_conv_features1, shape=[-1, sequences_count1 * features_size1])
        self.output_features2 = tf.reshape(self.multi_conv_features2, shape=[-1, sequences_count2 * features_size2])

        self.output_features = tf.concat([self.output_features1, self.output_features2], axis=1)

        self.output_features_add_numeric = tf.concat([self.output_features, self.input_x_3])

        with tf.variable_scope('output'):
            _, hidden_size = self.output_features_add_numeric.get_shape().as_list()
            w = tf.get_variable('weight', shape=[hidden_size, self.output_size],
                                initializer=tf.contrib.layers.xavier_initializer())
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w)
            g = tf.get_variable("g", initializer=tf.sqrt(1.0 / hidden_size * self.output_size))
            b = tf.get_variable("bias", shape=[self.output_size], initializer=tf.zeros_initializer)
            normed_w = g * w * tf.rsqrt(tf.reduce_sum(tf.square(w)))

            self.output_y = tf.matmul(self.output_features_add_numeric, normed_w) + b
            self.output_y = tf.nn.relu(self.output_y)

    def get_align_weights(self, source_state, scope):
        with tf.variable_scope(scope + "_" + "align_weight"):
            _, sequences_count, hidden_size = source_state.get_shape().as_list()
            v = tf.get_variable('v_a', shape=[hidden_size, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
            g = tf.get_variable("g", initializer=tf.sqrt(1.0 / hidden_size))
            b = tf.get_variable("bias", shape=[hidden_size], initializer=tf.zeros_initializer)
            # activation
            align_weights_logits = tf.nn.tanh(source_state + b)
            # transform [batch_size * sequences_count, hidden_size]
            align_weights_logits = tf.reshape(align_weights_logits, (-1, hidden_size))
            normed_v = g * v * tf.rsqrt(tf.reduce_sum(tf.square(v)))
            align_weights = tf.reshape(tf.matmul(align_weights_logits, normed_v), (-1, sequences_count))
            align_weights = tf.expand_dims(align_weights, axis=-1)
            # normalized
            align_weights_max = tf.reduce_max(align_weights, axis=1, keep_dims=True)
            align_weights = tf.nn.softmax(align_weights - align_weights_max)
        # [batch_size * sequences_count]
        return align_weights

    def compute_loss(self):
        with tf.name_scope("loss"):
            # 正则项loss
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'projection' in v.name])
            l2_losses += tf.add_n([tf.nn.l2_loss(v) for v in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)])
            # 对数后取欧氏距离
            self.core_loss = tf.reduce_mean(tf.square(tf.log(self.input_y + 1.0) - tf.log(self.output_y + 1.0)))
            self.l2_losses = l2_losses + self.l2_lambda

            return self.core_loss + l2_losses * self.l2_lambda

    def train(self):
        # 学习率(自我递减)
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                   self.decay_steps, self.decay_rate, staircase=True)
        self.learning_rate = learning_rate

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.contrib.layers.optimize_loss(self.loss,
                                                       global_step=self.global_step,
                                                       learning_rate=self.learning_rate,
                                                       optimizer='Adam',
                                                       clip_gradients=self.clip_gradients)

        return train_op

    def instantiate_word_embedding(self):
        with tf.device('/cpu:0'):
            self.word_embedding_mat = tf.get_variable('embedding',
                                                      shape=[self.vocabulary_size, self.word_embedding_size],
                                                  initializer=tf.constant_initializer(self.word_embedding_mat, dtype=tf.float32))

    def embedding_lookup_subword(self, embedding_mat, inputs_id, inputs_ratio):
        input_embed = tf.nn.embedding_lookup(embedding_mat, inputs_id)
        #[None, sequences_count, sequence_length, subword_length, embedding_dim]
        #[None, sequences_count, sequence_length, subword_length, 1]
        # ---> [None, sequences_count, sequence_length, embedding_dim]
        norm_embed = tf.reduce_sum(tf.multiply(input_embed, tf.expand_dims(inputs_ratio, axis=-1)), axis=-2)
        return norm_embed


    def train_step(self, x_1_batch, x_2_batch, x_1_ratio_batch, x_2_ratio_batch, x_3_train_batch, y_batch, dropout_keep_prob):
        feed_dict = {
            self.input_x_1: x_1_batch,
            self.input_x_2: x_2_batch,
            self.input_x_1_ratio: x_1_ratio_batch,
            self.input_x_2_ratio: x_2_ratio_batch,
            self.input_x_3: x_3_train_batch,
            self.input_y: y_batch,
            self.is_training: True,
            self.dropout_keep_prob: dropout_keep_prob
        }
        loss, _, _ = self.sess.run([self.loss, self.train_op, self.global_step],
                              feed_dict=feed_dict)
        return loss

    def dev_step(self, x_1_batch, x_2_batch, x_1_ratio_batch, x_2_ratio_batch, x_3_train_batch, y_batch):
        feed_dict = {
            self.input_x_1: x_1_batch,
            self.input_x_2: x_2_batch,
            self.input_x_1_ratio: x_1_ratio_batch,
            self.input_x_2_ratio: x_2_ratio_batch,
            self.input_x_3: x_3_train_batch,
            self.input_y: y_batch,
            self.is_training: False,
            self.dropout_keep_prob: 1.0
        }
        core_loss = self.sess.run([self.core_loss],
                             feed_dict=feed_dict)[0]
        return core_loss

    def predict_step(self, x_1_batch, x_2_batch, x_1_ratio_batch, x_2_ratio_batch, x_3_train_batch):
        feed_dict = {
            self.input_x_1: x_1_batch,
            self.input_x_2: x_2_batch,
            self.input_x_1_ratio: x_1_ratio_batch,
            self.input_x_2_ratio: x_2_ratio_batch,
            self.input_x_3: x_3_train_batch,
            self.is_training: False,
            self.dropout_keep_prob: 1.0
        }
        output_y = self.sess.run([self.output_y],
                            feed_dict=feed_dict)[0]
        return output_y

    def predict_output_features_step(self, x_1_batch, x_2_batch, x_1_ratio_batch, x_2_ratio_batch, x_3_train_batch):
        feed_dict = {
            self.input_x_1: x_1_batch,
            self.input_x_2: x_2_batch,
            self.input_x_1_ratio: x_1_ratio_batch,
            self.input_x_2_ratio: x_2_ratio_batch,
            self.input_x_3: x_3_train_batch,
            self.is_training: False,
            self.dropout_keep_prob: 1.0
        }
        output_features = self.sess.run([self.output_features],
                                   feed_dict=feed_dict)[0]
        return output_features

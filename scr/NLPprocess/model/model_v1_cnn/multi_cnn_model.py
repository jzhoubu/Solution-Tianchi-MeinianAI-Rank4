import tensorflow as tf

from model.utils import leaky_relu


class MultiCnnModel():
    def __init__(self, batch_size, sequences_count, sequence_length, word_embedding_size, word_embedding_mat,
                 vocabulary_size, conv_filter_sizes, num_filters, output_size, hidden_size, is_training, dropout_keep_prob=0.5,
                 l2_lambda=0.0001, learning_rate=0.001, decay_steps=1000, decay_rate=0.9, clip_gradients=3.0):

        # 每个batch的大小
        self.batch_size = batch_size
        # 子句的数量
        self.sequences_count = sequences_count
        # 句子的长度
        self.sequence_length = sequence_length
        # 词向量维度
        self.word_embedding_size = word_embedding_size
        # 词向量矩阵
        self.word_embedding_mat = word_embedding_mat
        # 词典size
        self.vocabulary_size = vocabulary_size
        # 卷积层filter_size
        self.conv_filter_sizes = conv_filter_sizes
        # 输出通道数
        self.num_filters = num_filters
        # 总输出通道数
        self.num_filters_total = num_filters * len(self.conv_filter_sizes)
        # 输出dense layer维度 #TODO 后续可能用到, 目前最后特征层的输出维度为 num_filters_total
        self.hidden_size = hidden_size
        # 输出维度
        self.output_size = output_size
        # dropout
        self.dropout_keep_prob_value = dropout_keep_prob
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

        # 输入 [batch_size * sequences_count * sequence_length]
        self.input_x = tf.placeholder(tf.int32, [None, self.sequences_count, self.sequence_length])
        # 输出
        self.input_y = tf.placeholder(tf.float32, [None, self.output_size])
        # global_step
        self.global_step = tf.Variable(0, trainable=False, name='Global_step')

        self.instantiate_word_embedding()

        # cnn层
        # self.multi_conv_features = self.multi_conv_layer()
        self.multi_conv_features = self.multi_conv_layer()
        # 输出inference层
        self.inference()

        # loss function计算
        self.loss = self.compute_loss()
        # train
        self.train_op = self.train()

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth=True

        self.sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def conv_layer(self, sequence, scope='conv_layer'):
        """
            cnn层, 每列的数据共用同样的cnn层
        :param sequence:
        :param scope:
        :return:
        """
        pooled_outputs = []
        for i, filter_size in enumerate(self.conv_filter_sizes):
            with tf.variable_scope(scope + '-%s' % filter_size, reuse=tf.AUTO_REUSE):
                conv_filter = tf.get_variable('filter-%s' % filter_size, [filter_size, self.word_embedding_size, 1, self.num_filters])
                # [batch_size * sequence_length * word_embedding_size * 1]
                conv = tf.nn.conv2d(sequence, conv_filter, strides=[1,1,1,1], padding='VALID', name='conv')
                # [batch_size * (sequence_length - filter_size + 1) * 1 * num_filters]
                bias = tf.get_variable('conv_bias-%s' % filter_size, [self.num_filters])
                h = leaky_relu(tf.nn.bias_add(conv, bias))
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1], strides=[1,1,1,1], padding='VALID', name='pool')
                
                # [batch_size * 1 * 1 * num_filters]
                pooled_outputs.append(pooled)

        # [batch_size * 1 * 1 * num_filters_total]
        self.h_pool = tf.concat(pooled_outputs, -1)
        # [batch_size * num_filters_total]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])

        with tf.name_scope('dropout'):
            conv_feature = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)

        with tf.variable_scope(scope + '_projection', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('weight', shape=[self.num_filters_total, self.hidden_size],
                                initializer=tf.contrib.layers.xavier_initializer())
            g = tf.get_variable("g", initializer=tf.sqrt(1.0 / self.hidden_size))
            b = tf.get_variable("bias", shape=[self.hidden_size], initializer=tf.zeros_initializer)
            normed_v = g * w * tf.rsqrt(tf.reduce_sum(tf.square(w)))
            conv_feature = tf.matmul(conv_feature, normed_v) + b
            # conv_feature = tf.matmul(conv_feature, w) + b

        return conv_feature

    def multi_conv_layer(self):
        # [batch_size * sequences_count * sequence_length * word_embedding_size]
        embedded_input_x = tf.nn.embedding_lookup(self.word_embedding_mat, self.input_x)
        # [batch_size * sequences_count * word_embedding_size]
        # flatted_input_x = tf.reduce_mean(embedded_input_x, axis=2, keep_dims=False)
        # flatted_input_x = tf.reshape(embedded_input_x, shape=[-1, self.sequences_count, self.sequence_length * self.word_embedding_size])
        # [batch_size * sequences_count * 1]
        # aligned_weight = self.get_align_weights(flatted_input_x)

        with tf.name_scope('multi_conv_layer'):
            # TODO highway features
            conv_features = []
            sub_sequences = tf.split(embedded_input_x, self.sequences_count, axis=1)
            for i, sub_sequence in enumerate(sub_sequences):
                sub_sequence = tf.squeeze(sub_sequence, axis=1)
                sub_sequence = tf.expand_dims(sub_sequence, axis=-1)
                sub_conv_feature = self.conv_layer(sub_sequence)
                conv_features.append(sub_conv_feature)

            # [batch_size * num_filters_total] * sequence_counts --> [batch_size * sequences_count * num_filters_total]
            multi_conv_features = tf.stack(conv_features, axis=1)
            
            with tf.name_scope('dropout'):
                multi_conv_features = tf.nn.dropout(multi_conv_features, self.dropout_keep_prob)

            return multi_conv_features

    def inference(self):

        _, sequences_count, features_size = self.multi_conv_features.get_shape().as_list()

        self.output_features = tf.reshape(self.multi_conv_features, shape=[-1, sequences_count * features_size])

        self.output_y = tf.layers.dense(self.output_features,
                                        self.output_size,
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        name='projection')
    
    def get_align_weights(self, source_state):
        with tf.variable_scope('align_weight'):
            _, sequences_count, hidden_size = source_state.get_shape().as_list()
            v = tf.get_variable('v_a', shape=[hidden_size, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
            g = tf.get_variable("align_g", initializer=tf.sqrt(1.0/hidden_size))
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
        #[batch_size * sequences_count]
        return align_weights
        
    def compute_loss(self):
        with tf.name_scope("loss"):
            # 正则项loss
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'projection' in v.name])
            # 对数后取欧氏距离
            self.core_loss = tf.reduce_mean(tf.square(tf.log(self.input_y + 1.0) - tf.log(self.output_y + 1.0)))

            return self.core_loss + l2_losses * self.l2_lambda
    #
    def train(self):
        # 学习率(自我递减)
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                   self.decay_steps, self.decay_rate, staircase=True)
        self.learning_rate = learning_rate

        train_op = tf.contrib.layers.optimize_loss(self.loss, global_step=self.global_step,
                                                   learning_rate=self.learning_rate, optimizer='Adam',
                                                   clip_gradients=self.clip_gradients)
        return train_op

    def instantiate_word_embedding(self):
        with tf.device('/cpu:0'):
            self.word_embedding_mat = tf.get_variable('embedding',
                                                      shape=[self.vocabulary_size, self.word_embedding_size],
                                         initializer=tf.constant_initializer(self.word_embedding_mat, dtype=tf.float32))

    def train_step(self, x_batch, y_batch):
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.is_training: True,
            self.dropout_keep_prob: self.dropout_keep_prob_value
        }
        loss, _, _ = self.sess.run([self.loss, self.train_op, self.global_step],
                      feed_dict=feed_dict)
        return loss

    def dev_step(self, x_batch, y_batch):
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.is_training: False,
            self.dropout_keep_prob: 1.0
        }
        core_loss = self.sess.run([self.core_loss],
                                  feed_dict=feed_dict)[0]
        return core_loss

    def predict_step(self, x_batch):
        feed_dict = {
            self.input_x: x_batch,
            self.is_training: False,
            self.dropout_keep_prob: 1.0
        }
        output_y = self.sess.run([self.output_y],
                                  feed_dict=feed_dict)[0]
        return output_y

    def predict_output_features_step(self, x_batch):
        feed_dict = {
            self.input_x: x_batch,
            self.is_training: False,
            self.dropout_keep_prob: 1.0
        }
        output_features = self.sess.run([self.output_features],
                                  feed_dict=feed_dict)[0]
        return output_features
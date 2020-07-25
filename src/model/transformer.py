# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/23
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/7/25
"""

import tensorflow as tf

from .module.utils_transformer import Encoder, Decoder, positional_encoding, create_padding_mask, create_look_ahead_mask
from .module.utils_nn import CustomSchedule, get_sparse_softmax_cross_entropy_loss, get_accuracy
from .module.beam_search import sequence_greedy_search, sequence_beam_search


class Transformer:
    def __init__(self, config):
        self.pad_id = config.pad_id
        self.sos_id = config.sos_id
        self.eos_id = config.eos_id
        self.src_vocab_size = config.src_vocab_size
        self.tgt_vocab_size = config.tgt_vocab_size
        self.max_seq_length = config.max_seq_length
        self.beam_size = config.top_k
        self.beam_search = config.beam_search

        self.word_em_size = config.word_em_size
        self.num_layer = config.num_layer
        self.num_head = config.num_head
        self.model_dim = config.model_dim
        self.fc_size = config.fc_size_m

        self.lr = config.lr
        self.dropout = config.dropout

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.src = tf.placeholder(tf.int32, [None, None], name='src')
        self.tgt = tf.placeholder(tf.int32, [None, None], name='tgt')
        self.src_len = tf.placeholder(tf.int32, [None], name='src_len')
        self.tgt_len = tf.placeholder(tf.int32, [None], name='tgt_len')
        self.training = tf.placeholder(tf.bool, [], name='training')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.src_embedding = tf.keras.layers.Embedding(self.src_vocab_size, self.word_em_size, name='src_embedding')
        self.tgt_embedding = tf.keras.layers.Embedding(self.tgt_vocab_size, self.word_em_size, name='tgt_embedding')
        self.pos_embedding = positional_encoding(5000, self.model_dim)
        self.src_em_dropout = tf.keras.layers.Dropout(self.dropout)
        self.tgt_em_dropout = tf.keras.layers.Dropout(self.dropout)
        self.encoder = Encoder(self.num_layer, self.model_dim, self.num_head, self.fc_size, self.dropout)
        self.decoder = Decoder(self.num_layer, self.model_dim, self.num_head, self.fc_size, self.dropout)
        self.final_dense = tf.keras.layers.Dense(self.tgt_vocab_size, name='final_dense')

        if config.optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr)
        elif config.optimizer == 'Adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif config.optimizer == 'Adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.lr)
        elif config.optimizer == 'SGD':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif config.optimizer == 'custom':
            # Adam optimizer with a custom learning rate scheduler according to the formula in the paper
            self.lr = CustomSchedule(self.model_dim, self.global_step)
            self.optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.98, epsilon=1e-9)
        else:
            assert False

        logits, self.predicted_ids = self.forward()
        self.loss = get_sparse_softmax_cross_entropy_loss(self.tgt[:, 1:], logits, self.tgt_len - 1)
        self.accu = get_accuracy(self.tgt[:, 1:], logits, self.tgt_len - 1)
        self.gradients, self.train_op = self.get_train_op()

        tf.summary.scalar('learning_rate', self.lr() if callable(self.lr) else self.lr)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accu)
        self.summary = tf.summary.merge_all()

    def forward(self):
        # Add word embedding and position embedding.
        src_em, tgt_em = self.embedding_layer()

        # Used in the 1st attention block in the encoder and the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder inputs and outputs.
        padding_mask = create_padding_mask(self.src, self.pad_id)
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by the decoder.
        look_ahead_mask = create_look_ahead_mask(self.tgt[:, :-1], self.pad_id)

        enc_output = self.encoding_layer(src_em, padding_mask)

        logits = self.training_decoding_layer(tgt_em[:, :-1], enc_output, look_ahead_mask, padding_mask)
        predicted_ids = self.inference_decoding_layer(enc_output, padding_mask)

        return logits, predicted_ids

    def get_train_op(self):
        gradients = tf.gradients(self.loss, tf.trainable_variables())
        gradients, _ = tf.clip_by_global_norm(gradients, 5)
        train_op = self.optimizer.apply_gradients(zip(gradients, tf.trainable_variables()), self.global_step)

        return gradients, train_op

    def embedding_layer(self):
        with tf.device('/cpu:0'):
            src_em = self.src_embedding(self.src)
            tgt_em = self.tgt_embedding(self.tgt)

        src_em *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        src_em += self.pos_embedding[:, :tf.shape(src_em)[1], :]
        src_em = self.src_em_dropout(src_em, training=self.training)

        tgt_em *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        tgt_em += self.pos_embedding[:, :tf.shape(tgt_em)[1], :]
        tgt_em = self.tgt_em_dropout(tgt_em, training=self.training)

        return src_em, tgt_em

    def encoding_layer(self, src_em, padding_mask):
        # enc_output.shape == (batch_size, src_seq_len, d_model)
        enc_output = self.encoder(src_em, padding_mask, self.training)

        return enc_output

    def training_decoding_layer(self, tgt_em, enc_output, look_ahead_mask, padding_mask):
        # dec_output.shape == (batch_size, tgt_seq_len, d_model)
        dec_output, _ = self.decoder(tgt_em, enc_output, look_ahead_mask, padding_mask, self.training)

        logits = self.final_dense(dec_output)  # (batch_size, tgt_seq_len, target_vocab_size)

        return logits

    def inference_decoding_layer(self, enc_output, padding_mask):
        """Return predicted sequence."""
        # Create initial ids and cache.
        initial_ids = tf.fill([self.batch_size], self.sos_id)
        initial_cache = {
            'layer{}'.format(i + 1): {
                'k': tf.zeros([self.batch_size, 0, self.model_dim]),
                'v': tf.zeros([self.batch_size, 0, self.model_dim]),
            } for i in range(self.num_layer)
        }
        initial_cache['enc_outputs'] = enc_output
        initial_cache['padding_mask'] = padding_mask

        if not self.beam_search:
            # Use greedy search to find the decoded sequence and score.
            predicted_ids, _ = sequence_greedy_search(
                symbols_to_logits_fn=self._symbols_to_logits_fn,
                initial_ids=initial_ids,
                initial_cache=initial_cache,
                max_decode_length=self.max_seq_length,
                eos_id=self.eos_id,
            )
        else:
            # Use beam search to find the top beam_size sequences and scores.
            predicted_ids, _ = sequence_beam_search(
                symbols_to_logits_fn=self._symbols_to_logits_fn,
                initial_ids=initial_ids,
                initial_cache=initial_cache,
                vocab_size=self.tgt_vocab_size,
                beam_size=self.beam_size,
                alpha=0.0,
                max_decode_length=self.max_seq_length,
                eos_id=self.eos_id,
            )

            # Get the top sequence for each batch element.
            predicted_ids = predicted_ids[:, 0, :]

        return predicted_ids[:, 1:]

    def _symbols_to_logits_fn(self, step, decoded_ids, cache):
        """
        Go from ids to logits for next symbol.

        :param step: Current loop step.
        :param decoded_ids: Decoded sequences.
        :param cache: Dictionary of cached values.
        :return:
        """
        look_ahead_mask = create_look_ahead_mask(decoded_ids, self.pad_id)
        look_ahead_mask = look_ahead_mask[:, :, step:step + 1, :step + 1]

        current_id = decoded_ids[:, step:step + 1]
        current_id_em = self.tgt_embedding(current_id)
        current_id_em *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        current_id_em += self.pos_embedding[:, step:step + 1, :]

        dec_output, _ = self.decoder(
            current_id_em,
            cache['enc_outputs'],
            look_ahead_mask,
            cache['padding_mask'],
            self.training,
            cache
        )

        logits = self.final_dense(dec_output)
        logits = tf.squeeze(logits, axis=[1])  # (batch_size, target_vocab_size)

        return logits, cache

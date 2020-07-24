# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/23
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/7/24
"""

import tensorflow as tf

from .module.utils_transformer import Encoder, Decoder, create_padding_mask, create_look_ahead_mask
from .module.utils_nn import CustomSchedule, get_sparse_softmax_cross_entropy_loss, get_accuracy
from .module.beam_search import sequence_beam_search


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
        self.encoder = Encoder(self.src_embedding, self.num_layer, self.model_dim, self.num_head, self.fc_size,
                               self.src_vocab_size, self.dropout)
        self.decoder = Decoder(self.tgt_embedding, self.num_layer, self.model_dim, self.num_head, self.fc_size,
                               self.tgt_vocab_size, self.dropout)
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
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(self.src, self.pad_id)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(self.src, self.pad_id)

        enc_output = self.encoding_layer(self.src, enc_padding_mask)

        logits = self.training_decoding_layer(self.tgt[:, :-1], enc_output, dec_padding_mask)

        predicted_ids = self.inference_decoding_layer(enc_output, dec_padding_mask)

        return logits, predicted_ids

    def get_train_op(self):
        gradients = tf.gradients(self.loss, tf.trainable_variables())
        train_op = self.optimizer.apply_gradients(zip(gradients, tf.trainable_variables()), self.global_step)

        return gradients, train_op

    def encoding_layer(self, src, enc_padding_mask):
        # enc_output.shape == (batch_size, src_seq_len, d_model)
        enc_output = self.encoder(src, enc_padding_mask, self.training)

        return enc_output

    def training_decoding_layer(self, tgt, enc_output, dec_padding_mask):
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by the decoder.
        look_ahead_mask = create_look_ahead_mask(tgt, self.pad_id)

        # dec_output.shape == (batch_size, tgt_seq_len, d_model)
        dec_output, _ = self.decoder(tgt, enc_output, look_ahead_mask, dec_padding_mask, self.training)

        logits = self.final_dense(dec_output)  # (batch_size, tgt_seq_len, target_vocab_size)

        return logits

    def inference_decoding_layer(self, enc_output, dec_padding_mask):
        """Return predicted sequence."""

        def _symbols_to_logits_fn(tgt):
            look_ahead_mask = create_look_ahead_mask(tgt, self.pad_id)
            dec_output, _ = self.decoder(tgt, enc_output, look_ahead_mask, dec_padding_mask, self.training)
            logits = self.final_dense(dec_output)

            return logits[:, -1, :]

        # Use beam search to find the top beam_size sequences and scores.
        predicted_ids, _ = sequence_beam_search(
            symbols_to_logits_fn=_symbols_to_logits_fn,
            initial_ids=tf.fill([self.batch_size], self.sos_id),
            vocab_size=self.tgt_vocab_size,
            beam_size=self.beam_size if self.beam_search else 1,
            alpha=0.0,
            max_decode_length=self.max_seq_length,
            eos_id=self.eos_id,
        )

        # Get the top sequence for each batch element
        predicted_ids = predicted_ids[:, 0, 1:]

        return predicted_ids

# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/2/20
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/7/24
"""

import tensorflow as tf


class CustomSchedule:
    def __init__(self, d_model, global_step, warmup_steps=4000):
        self.d_model = tf.cast(d_model, tf.float32)
        self.global_step = tf.cast(global_step, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self):
        arg1 = tf.math.rsqrt(self.global_step)
        arg2 = self.global_step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def get_sparse_softmax_cross_entropy_loss(labels, logits, mask_sequence_length=None):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    if mask_sequence_length is not None:
        mask = tf.sequence_mask(mask_sequence_length, dtype=tf.float32)
        loss = tf.reduce_mean(tf.reduce_sum(mask * loss, axis=-1) / tf.reduce_sum(mask, axis=-1))
    else:
        loss = tf.reduce_mean(loss)

    return loss


def get_sparse_cross_entropy_loss(y_true, y_pred, mask_sequence_length=None):
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    if mask_sequence_length is not None:
        mask = tf.sequence_mask(mask_sequence_length, dtype=tf.float32)
        loss = tf.reduce_mean(tf.reduce_sum(mask * loss, axis=-1) / tf.reduce_sum(mask, axis=-1))
    else:
        loss = tf.reduce_mean(loss)

    return loss


def get_accuracy(y_true, y_pred, mask_sequence_length=None):
    pred_ids = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
    accuracy = tf.cast(tf.equal(y_true, pred_ids), tf.float32)
    if mask_sequence_length is not None:
        mask = tf.sequence_mask(mask_sequence_length, dtype=tf.float32)
        accuracy = tf.reduce_mean(tf.reduce_sum(mask * accuracy, axis=-1) / tf.reduce_sum(mask, axis=-1))
    else:
        accuracy = tf.reduce_mean(accuracy)

    return accuracy

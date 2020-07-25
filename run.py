# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/23
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2020/7/25
"""

import os
import json
import logging
import argparse
import tensorflow as tf
from tqdm import tqdm
from collections import defaultdict

from src.config import Config
from src.data_reader import DataReader
from src.evaluator import Evaluator
from src.model import get_model
from src.utils import init_logger, log_title, read_json_dict, read_json_lines, save_json, \
    convert_list, make_batch_iter, make_batch_data

logger = logging.getLogger()

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, required=True)
parser.add_argument('--do_train', action='store_true', default=False)
parser.add_argument('--do_eval', action='store_true', default=False)
parser.add_argument('--do_test', action='store_true', default=False)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--max_seq_length', type=int, default=128)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--log_steps', type=int, default=100)
parser.add_argument('--save_steps', type=int, default=500)
parser.add_argument('--pre_train_epochs', type=int, default=0)
parser.add_argument('--early_stop', type=int, default=0)
parser.add_argument('--model_file', type=str)
parser.add_argument('--beam_search', action='store_true', default=False)
args = parser.parse_args()

config = Config('.', args.model,
                num_epoch=args.epoch, batch_size=args.batch,
                max_seq_length=args.max_seq_length,
                optimizer=args.optimizer, lr=args.lr, dropout=args.dropout,
                beam_search=args.beam_search)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True


def save_outputs(predicted_ids, id_2_label, input_file, output_file):
    golden_outputs = []
    for line in read_json_lines(input_file):
        golden_outputs.append(line['tgt'])

    with open(output_file, 'w', encoding='utf-8') as fout:
        for tgt, golden in zip(predicted_ids, golden_outputs):
            tgt[-1] = config.eos_id
            tgt = convert_list(tgt[:tgt.index(config.eos_id)], id_2_label, config.pad, config.unk)
            print(json.dumps({'tgt': ' '.join(tgt), 'golden': golden}, ensure_ascii=False), file=fout)


def run_test(sess, model, test_data):
    predicted_ids = []
    batch_iter = tqdm(list(make_batch_iter(test_data, config.batch_size, shuffle=False)), desc='Test')
    for batch in batch_iter:
        src, src_len, _, _ = make_batch_data(batch, config.pad_id)

        _predicted_ids = sess.run(
            model.predicted_ids,
            feed_dict={
                model.batch_size: len(src),
                model.src: src,
                model.src_len: src_len,
                model.training: False
            }
        )
        predicted_ids.extend(_predicted_ids.tolist())

    return predicted_ids


def run_evaluate(sess, model, valid_data):
    steps = 0
    predicted_ids = []
    total_loss = 0.0
    total_accu = 0.0
    batch_iter = tqdm(list(make_batch_iter(valid_data, config.batch_size, shuffle=False)), desc='Eval')
    for batch in batch_iter:
        src, src_len, tgt, tgt_len = make_batch_data(batch, config.pad_id)

        _predicted_ids, loss, accu, global_step, summary = sess.run(
            [model.predicted_ids, model.loss, model.accu, model.global_step, model.summary],
            feed_dict={
                model.batch_size: len(src),
                model.src: src,
                model.src_len: src_len,
                model.tgt: tgt,
                model.tgt_len: tgt_len,
                model.training: False
            }
        )
        predicted_ids.extend(_predicted_ids.tolist())

        steps += 1
        total_loss += loss
        total_accu += accu
        batch_iter.set_description('Eval: loss is {:>.4f}, accuracy is {:>.4f}'.format(loss, accu))

    return predicted_ids, total_loss / steps, total_accu / steps


def run_train(sess, model, train_data, valid_data, saver, evaluator, summary_writer=None):
    flag = 0
    best_valid_result = 0.0
    valid_log_history = defaultdict(list)
    global_step = 0
    for i in range(config.num_epoch):
        logger.info(log_title('Train Epoch: {}'.format(i + 1)))
        steps = 0
        total_loss = 0.0
        total_accu = 0.0
        batch_iter = tqdm(list(make_batch_iter(train_data, config.batch_size, shuffle=True)), desc='Train')
        for batch in batch_iter:
            src, src_len, tgt, tgt_len = make_batch_data(batch, config.pad_id)

            _, loss, accu, global_step, summary = sess.run(
                [model.train_op, model.loss, model.accu, model.global_step, model.summary],
                feed_dict={
                    model.batch_size: len(src),
                    model.src: src,
                    model.src_len: src_len,
                    model.tgt: tgt,
                    model.tgt_len: tgt_len,
                    model.training: True
                }
            )

            steps += 1
            total_loss += loss
            total_accu += accu
            batch_iter.set_description('Train: loss is {:>.4f}, accuracy is {:>.4f}'.format(loss, accu))
            if global_step % args.log_steps == 0 and summary_writer is not None:
                summary_writer.add_summary(summary, global_step)
            if global_step % args.save_steps == 0:
                # evaluate saved models after pre-train epochs
                if i < args.pre_train_epochs:
                    logger.info('saving model-{}'.format(global_step))
                    saver.save(sess, config.model_file, global_step=global_step)
                else:
                    predicted_ids, valid_loss, valid_accu = run_evaluate(sess, model, valid_data)
                    logger.info('valid loss is {:>.4f}, valid accuracy is {:>.4f}'.format(valid_loss, valid_accu))

                    save_outputs(predicted_ids, config.id_2_tgt, config.valid_data, config.valid_outputs)
                    valid_results = evaluator.evaluate(config.valid_data, config.valid_outputs, config.to_lower)

                    # early stop
                    if valid_results['BLEU 4'] >= best_valid_result:
                        flag = 0
                        best_valid_result = valid_results['BLEU 4']
                        logger.info('saving model-{}'.format(global_step))
                        saver.save(sess, config.model_file, global_step=global_step)
                        save_json(valid_results, config.valid_results)
                    elif flag < args.early_stop:
                        flag += 1
                    elif args.early_stop:
                        return valid_log_history

                    for key, value in valid_results.items():
                        valid_log_history[key].append(value)
                    valid_log_history['loss'].append(valid_loss)
                    valid_log_history['accuracy'].append(valid_accu)
                    valid_log_history['global_step'].append(int(global_step))
        logger.info('train loss is {:>.4f}, train accuracy is {:>.4f}'.format(total_loss / steps, total_accu / steps))
    saver.save(sess, config.model_file, global_step=global_step)

    return valid_log_history


def main():
    os.makedirs(config.temp_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    os.makedirs(config.train_log_dir, exist_ok=True)

    logger.setLevel(logging.INFO)
    init_logger(logging.INFO)

    logger.info('loading dict...')
    config.src_2_id, config.id_2_src = read_json_dict(config.src_vocab_dict)
    config.src_vocab_size = min(config.src_vocab_size, len(config.src_2_id))
    config.tgt_2_id, config.id_2_tgt = read_json_dict(config.tgt_vocab_dict)
    config.tgt_vocab_size = min(config.tgt_vocab_size, len(config.tgt_2_id))

    data_reader = DataReader(config)
    evaluator = Evaluator('tgt')

    logger.info('building model...')
    model = get_model(config)
    saver = tf.train.Saver(max_to_keep=10)

    if args.do_train:
        logger.info('loading data...')
        train_data = data_reader.load_train_data()
        valid_data = data_reader.load_valid_data()

        logger.info(log_title('Trainable Variables'))
        for v in tf.trainable_variables():
            logger.info(v)

        logger.info(log_title('Gradients'))
        for g in model.gradients:
            logger.info(g)

        with tf.Session(config=sess_config) as sess:
            model_file = args.model_file
            if model_file is None:
                model_file = tf.train.latest_checkpoint(os.path.join(config.result_dir, config.current_model))
            if model_file is not None:
                logger.info('loading model from {}...'.format(model_file))
                saver.restore(sess, model_file)
            else:
                logger.info('initializing from scratch...')
                tf.global_variables_initializer().run()

            train_writer = tf.summary.FileWriter(config.train_log_dir, sess.graph)
            valid_log_history = run_train(sess, model, train_data, valid_data, saver, evaluator, train_writer)
            save_json(valid_log_history, os.path.join(config.result_dir, config.current_model, 'valid_log_history.json'))

    if args.do_eval:
        logger.info('loading data...')
        valid_data = data_reader.load_valid_data()

        with tf.Session(config=sess_config) as sess:
            model_file = args.model_file
            if model_file is None:
                model_file = tf.train.latest_checkpoint(os.path.join(config.result_dir, config.current_model))
            if model_file is not None:
                logger.info('loading model from {}...'.format(model_file))
                saver.restore(sess, model_file)

                predicted_ids, valid_loss, valid_accu = run_evaluate(sess, model, valid_data)
                logger.info('average valid loss: {:>.4f}, average valid accuracy: {:>.4f}'.format(valid_loss, valid_accu))

                logger.info(log_title('Saving Result'))
                save_outputs(predicted_ids, config.id_2_tgt, config.valid_data, config.valid_outputs)
                results = evaluator.evaluate(config.valid_data, config.valid_outputs,config.to_lower)
                save_json(results, config.valid_results)
            else:
                logger.info('model not found!')

    if args.do_test:
        logger.info('loading data...')
        test_data = data_reader.load_test_data()

        with tf.Session(config=sess_config) as sess:
            model_file = args.model_file
            if model_file is None:
                model_file = tf.train.latest_checkpoint(os.path.join(config.result_dir, config.current_model))
            if model_file is not None:
                logger.info('loading model from {}...'.format(model_file))
                saver.restore(sess, model_file)

                predicted_ids = run_test(sess, model, test_data)

                logger.info(log_title('Saving Result'))
                save_outputs(predicted_ids, config.id_2_tgt, config.test_data, config.test_outputs)
                results = evaluator.evaluate(config.test_data, config.test_outputs, config.to_lower)
                save_json(results, config.test_results)
            else:
                logger.info('model not found!')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/2/20
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/7/24
"""

import logging
import collections
from tqdm import tqdm

from src.config import Config
from src.utils import init_logger, read_json_lines, save_json_dict

logger = logging.getLogger(__name__)


def build_dict(config):
    src_counter = collections.Counter()
    tgt_counter = collections.Counter()
    for line in tqdm(list(read_json_lines(config.train_data)), desc='Building dict'):
        src = line['src']
        tgt = line['tgt']
        if config.to_lower:
            src = src.lower()
            tgt = tgt.lower()
        for word in src.split():
            src_counter[word] += 1
        for word in tgt.split():
            tgt_counter[word] +=1

    src_counter[config.pad] = tgt_counter[config.pad] = 1e9 - config.pad_id
    src_counter[config.unk] = tgt_counter[config.unk] = 1e9 - config.unk_id
    src_counter[config.sos] = tgt_counter[config.sos] = 1e9 - config.sos_id
    src_counter[config.eos] = tgt_counter[config.eos] = 1e9 - config.eos_id
    src_counter[config.sep] = tgt_counter[config.sep] = 1e9 - config.sep_id
    src_counter[config.num] = tgt_counter[config.num] = 1e9 - config.num_id
    src_counter[config.time] = tgt_counter[config.time] = 1e9 - config.time_id
    logger.info('number of source words: {}'.format(len(src_counter)))
    logger.info('number of target words: {}'.format(len(tgt_counter)))

    word_dict = {}
    for word, _ in src_counter.most_common(config.src_vocab_size):
        word_dict[word] = len(word_dict)
    save_json_dict(word_dict, config.src_vocab_dict)

    word_dict = {}
    for word, _ in tgt_counter.most_common(config.tgt_vocab_size):
        word_dict[word] = len(word_dict)
    save_json_dict(word_dict, config.tgt_vocab_dict)


def preprocess():
    logger.setLevel(logging.INFO)
    init_logger(logging.INFO)

    config = Config('.', 'temp')

    logger.info('building dict...')
    build_dict(config)


if __name__ == '__main__':
    preprocess()

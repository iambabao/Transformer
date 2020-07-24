# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/2/20
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/7/24
"""

import os
import json
import copy
import pickle
import logging
from tqdm import tqdm

from src.utils import log_title, read_json_lines, convert_list

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, src, tgt=None):
        self.guid = guid
        self.src = src
        self.tgt = tgt

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    def __init__(self, guid, src_ids, tgt_ids):
        self.guid = guid
        self.src_ids = src_ids
        self.tgt_ids = tgt_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(examples, config):
    features = []
    for index, example in enumerate(tqdm(examples, desc='Converting Examples')):
        src_seq, tgt_seq = [], []

        for word in example.src.split():
            src_seq.append(word)

        if example.tgt:
            for word in example.tgt.split():
                tgt_seq.append(word)

        src_seq = [config.sos] + src_seq[:config.max_seq_length] + [config.eos]
        tgt_seq = [config.sos] + tgt_seq[:config.max_seq_length] + [config.eos]
        if config.to_lower:
            src_seq = list(map(str.lower, src_seq))
            tgt_seq = list(map(str.lower, tgt_seq))

        src_ids = convert_list(src_seq, config.src_2_id, config.pad_id, config.unk_id)
        tgt_ids = convert_list(tgt_seq, config.tgt_2_id, config.pad_id, config.unk_id)

        features.append(InputFeatures(example.guid, src_ids, tgt_ids))

        if index < 5:
            logger.info(log_title('Examples'))
            logger.info('guid: {}'.format(example.guid))
            logger.info('source input: {}'.format(src_seq))
            logger.info('source ids: {}'.format(src_ids))
            logger.info('target input: {}'.format(tgt_seq))
            logger.info('target ids: {}'.format(tgt_ids))

    return features


class DataReader:
    def __init__(self, config):
        self.config = config

    def load_train_data(self):
        if os.path.exists(self.config.cached_train_data):
            cached_data = pickle.load(open(self.config.cached_train_data, 'rb'))
            features = cached_data['features']
        else:
            examples, features = self._load_and_cache_data(self.config.train_data, self.config.cached_train_data)
        return features

    def load_valid_data(self):
        if os.path.exists(self.config.cached_valid_data):
            cached_data = pickle.load(open(self.config.cached_valid_data, 'rb'))
            features = cached_data['features']
        else:
            examples, features = self._load_and_cache_data(self.config.valid_data, self.config.cached_valid_data)
        return features

    def load_test_data(self):
        if os.path.exists(self.config.cached_test_data):
            cached_data = pickle.load(open(self.config.cached_test_data, 'rb'))
            features = cached_data['features']
        else:
            examples, features = self._load_and_cache_data(self.config.test_data, self.config.cached_test_data)
        return features

    def _load_and_cache_data(self, data_file, cache_file=None):
        examples = []
        for index, line in enumerate(tqdm(list(read_json_lines(data_file)), desc='Loading file: {}'.format(data_file))):
            src = line['src']
            tgt = line.get('tgt')
            examples.append(InputExample(index, src, tgt))

        features = convert_examples_to_features(examples, self.config)
        if cache_file:
            pickle.dump({'examples': examples, 'features': features}, open(cache_file, 'wb'))

        return examples, features

# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/2/20
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/7/24
"""

from .transformer import Transformer

model_list = {
    'transformer': Transformer,
}


def get_model(config):
    assert config.current_model in model_list

    return model_list[config.current_model](config)

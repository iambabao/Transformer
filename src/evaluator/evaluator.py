# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/2/20
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/6/30
"""

import logging
from nltk.translate.bleu_score import corpus_bleu

from src.utils import read_json_lines

logger = logging.getLogger(__name__)


def calc_bleu(references, hypotheses):
    list_of_references = [[v] for v in references]

    bleu1 = 100 * corpus_bleu(list_of_references, hypotheses, (1., 0., 0., 0.))
    bleu2 = 100 * corpus_bleu(list_of_references, hypotheses, (0.5, 0.5, 0., 0.))
    bleu3 = 100 * corpus_bleu(list_of_references, hypotheses, (0.33, 0.33, 0.33, 0.))
    bleu4 = 100 * corpus_bleu(list_of_references, hypotheses, (0.25, 0.25, 0.25, 0.25))

    return {'BLEU 1': bleu1, 'BLEU 2': bleu2, 'BLEU 3': bleu3, 'BLEU 4': bleu4}


class Evaluator:
    def __init__(self, key):
        self.key = key

    def evaluate(self, ref_file, hyp_file, to_lower):
        references = []
        for line in read_json_lines(ref_file):
            ref = line.get(self.key, '').strip().split()  # ref is a list of tokens
            if to_lower:
                ref = list(map(str.lower, ref))
            references.append(ref)

        hypotheses = []
        for line in read_json_lines(hyp_file):
            hyp = line.get(self.key, '').strip().split()  # hyp is a list of tokens
            if to_lower:
                hyp = list(map(str.lower, hyp))
            hypotheses.append(hyp)

        assert len(references) == len(hypotheses)

        results = {}
        results.update(calc_bleu(references, hypotheses))

        for key, value in results.items():
            logger.info('{}: {:>.4f}'.format(key, value))

        return results

"""
Sophisticated training noise.
"""

import hyperparameters
from vocabulary import wordmap

from common.myrandom import build
import sys

_indexed_weights = None
def indexed_weights():
    global _indexed_weights
    if _indexed_weights is not None:
        return _indexed_weights
    print >> sys.stderr, wordmap.len, "=?=", hyperparameters.VOCABULARY_SIZE
    assert wordmap.len == hyperparameters.VOCABULARY_SIZE
    if hyperparameters.NGRAM_FOR_TRAINING_NOISE == 0:
        _indexed_weights = [1 for id in range(wordmap.len)]
    elif hyperparameters.NGRAM_FOR_TRAINING_NOISE == 1:
        from common.json import load
        from common.file import myopen
        ngrams_file = hyperparameters.NGRAMS[(hyperparameters.NGRAM_FOR_TRAINING_NOISE, hyperparameters.VOCABULARY_SIZE)]
        print >> sys.stderr, "Reading ngrams from", ngrams_file, "..."
        from collections import defaultdict
        ngramcnt = defaultdict(int)
        for (ngram, cnt) in load(myopen(ngrams_file)):
            assert len(ngram) == 1
            ngramcnt[ngram[0]] = cnt + hyperparameters.TRAINING_NOISE_SMOOTHING_ADDITION
        _indexed_weights = [ngramcnt[wordmap.str(id)] for id in range(wordmap.len)]
        _indexed_weights = build(_indexed_weights)
    else: assert 0
    return _indexed_weights

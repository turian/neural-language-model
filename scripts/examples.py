"""
Methods for getting examples.
"""

from common.file import myopen
import string

import common.hyperparameters

def get_train_example():
    HYPERPARAMETERS = common.hyperparameters.read("language-model")

    from vocabulary import wordmap
    for l in myopen(HYPERPARAMETERS["TRAIN_SENTENCES"]):
        prevwords = []
        for w in string.split(l):
            w = string.strip(w)
            id = None
            if wordmap.exists(w):
                prevwords.append(wordmap.id(w))
                if len(prevwords) >= HYPERPARAMETERS["WINDOW_SIZE"]:
                    yield prevwords[-HYPERPARAMETERS["WINDOW_SIZE"]:]
            else:
                prevwords = []

def get_train_minibatch():
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    minibatch = []
    for e in get_train_example():
        minibatch.append(e)
        if len(minibatch) >= HYPERPARAMETERS["MINIBATCH SIZE"]:
            assert len(minibatch) == HYPERPARAMETERS["MINIBATCH SIZE"]
            yield minibatch
            minibatch = []

def get_validation_example():
    HYPERPARAMETERS = common.hyperparameters.read("language-model")

    from vocabulary import wordmap
    for l in myopen(HYPERPARAMETERS["VALIDATION_SENTENCES"]):
        prevwords = []
        for w in string.split(l):
            w = string.strip(w)
            if wordmap.exists(w):
                prevwords.append(wordmap.id(w))
                if len(prevwords) >= HYPERPARAMETERS["WINDOW_SIZE"]:
                    yield prevwords[-HYPERPARAMETERS["WINDOW_SIZE"]:]
            else:
                prevwords = []

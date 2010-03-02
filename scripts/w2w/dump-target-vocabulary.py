#!/usr/bin/env python
"""
Dump the w2w target vocabulary.
"""

import sys

if __name__ == "__main__":
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    HYPERPARAMETERS, options, args, newkeystr = common.options.reparse(HYPERPARAMETERS)
    import hyperparameters

    from common.mydict import sort as dictsort

    from vocabulary import wordmap
    from targetvocabulary import targetmap

    for w1 in wordmap().all:
        for l2 in targetmap()[w1]:
            print w1, l2, dictsort(targetmap()[w1][l2])

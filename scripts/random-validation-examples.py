#!/usr/bin/env python

import miscglobals
        
import string
import train
from vocabulary import wordmap

if __name__ == "__main__":
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    HYPERPARAMETERS, options, args = common.options.reparse(HYPERPARAMETERS)
    import hyperparameters

    ves = [e for e in train.get_validation_example()]
    import random
    random.shuffle(ves)
    for e in ves[:1000]:
        print string.join([wordmap.str(id) for id in e])

#!/usr/bin/env python

import hyperparameters, miscglobals
        
import string
import train
from vocabulary import wordmap

if __name__ == "__main__":
    import common.options
    hyperparameters.__dict__.update(common.options.reparse(hyperparameters.__dict__)[0])
    #miscglobals.__dict__ = common.options.reparse(miscglobals.__dict__)

    ves = [e for e in train.get_validation_example()]
    import random
    random.shuffle(ves)
    for e in ves[:1000]:
        print string.join([wordmap.str(id) for id in e])

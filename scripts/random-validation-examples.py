#!/usr/bin/env python
#
#  Print out validation examples, disregarding vocabulary.
#
#  @TODO: Don't duplicate get_example code here and twice in train.py
#

from common.file import myopen
import string
import sys

def get_example(f):
    import common.hyperparameters
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    for l in myopen(f):
        prevwords = []
        for w in string.split(l):
            w = string.strip(w)
            prevwords.append(w)
            if len(prevwords) >= HYPERPARAMETERS["WINDOW_SIZE"]:
                yield prevwords[-HYPERPARAMETERS["WINDOW_SIZE"]:]

if __name__ == "__main__":
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    HYPERPARAMETERS, options, args = common.options.reparse(HYPERPARAMETERS)
    import hyperparameters

    print >> sys.stderr, "Reading examples from %s" % HYPERPARAMETERS["ORIGINAL VALIDATION_SENTENCES"]
    ves = [e for e in get_example(HYPERPARAMETERS["ORIGINAL VALIDATION_SENTENCES"])]
    import random
    random.shuffle(ves)
    print >> sys.stderr, "Reading %d examples to %s" % (HYPERPARAMETERS["VALIDATION EXAMPLES"], HYPERPARAMETERS["VALIDATION_SENTENCES"])
    o = myopen(HYPERPARAMETERS["VALIDATION_SENTENCES"], "w")
    for e in ves[:HYPERPARAMETERS["VALIDATION EXAMPLES"]]:
        o.write(string.join(e) + "\n")

#!/usr/bin/env python
"""
Read in the w2w corpora (bi + monolingual), and build the vocabulary as
all words per language that occur at least HYPERPARAMETERS["W2W MINIMUM
WORD FREQUENCY"] times.
Each corpus is weighted in proportion to its length. (i.e. all words are equally weighted.)
"""

import sys
from common.stats import stats

def readwords(filename):
    print >> sys.stderr, "Processing %s" % filename
    i = 0
    for line in open(filename):
        i += 1
        if i % 100000 == 0:
            print >> sys.stderr, "Read line %d of %s..." % (i, filename)
            print >> sys.stderr, stats()
        for w in string.split(line):
            yield w

if __name__ == "__main__":
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    HYPERPARAMETERS, options, args, newkeystr = common.options.reparse(HYPERPARAMETERS)
    import hyperparameters

    import logging
    logging.basicConfig(level=logging.DEBUG)

    import w2w.corpora
    import string

    from collections import defaultdict
    wordfreq = defaultdict(int)
    for l1, l2, f1, f2, falign in w2w.corpora.bicorpora_filenames():
        for w in readwords(f1): wordfreq[(l1,w)] += 1
        for w in readwords(f2): wordfreq[(l2,w)] += 1

    for l, f in w2w.corpora.monocorpora_filenames():
        assert 0

    for (l, w) in wordfreq.keys():
        if wordfreq[(l, w)] < HYPERPARAMETERS["W2W MINIMUM WORD FREQUENCY"]:
            del wordfreq[(l, w)]
        if w == "*UNKNOWN*":
            del wordfreq[(l, w)]

    import w2w.vocabulary
    import common.idmap

    wordfreqkeys = [key for cnt, key in dictsort(wordfreq)]

#    for k in wordfreq.keys():
#        print k
    v = common.idmap.IDmap([(None, "*LBOUNDARY*"), (None, "*RBOUNDARY*")] + wordfreqkeys, allow_unknown=HYPERPARAMETERS["INCLUDE_UNKNOWN_WORD"], unknown_key=(None, "*UNKNOWN*"))
    w2w.vocabulary.write(v)

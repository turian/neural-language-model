#!/usr/bin/env python
"""
Read in the w2w corpora (bi + monolingual), and build the vocabulary as
all words per language that occur at least HYPERPARAMETERS["W2W MINIMUM
WORD FREQUENCY"] times.
Each corpus is weighted in proportion to its length. (i.e. all words are equally weighted.)
"""

import sys

def readwords(filename):
    print >> sys.stderr, "Processing %s" % filename
    for line in open(filename):
        for w in string.split(line):
            yield w

if __name__ == "__main__":
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    HYPERPARAMETERS, options, args, newkeystr = common.options.reparse(HYPERPARAMETERS)
    import hyperparameters

    import w2w.corpora
    import string

    from collections import defaultdict
    wordfreq = defaultdict(int)
    for l1, l2, f1, f2, falign in w2w.corpora.bicorpora_filenames():
        for w in readwords(f1): wordfreq[(l1,w)] += 1
        for w in readwords(f2): wordfreq[(l2,w)] += 1

    for l, f in w2w.corpora.monocorpora_filenames():
        assert 0

    for (l, w) in wordfreq:
        print l, w, wordfreq[(l, w)]

#    import vocabulary
#    import common.idmap
#
#    words = []
#
#    import string
#    for i, l in enumerate(common.file.myopen(HYPERPARAMETERS["MONOLINGUAL_VOCABULARY"])):
#        if HYPERPARAMETERS["INCLUDE_UNKNOWN_WORD"] and i+1 >= HYPERPARAMETERS["MONOLINGUAL_VOCABULARY_SIZE"]:
#            break
#        if not HYPERPARAMETERS["INCLUDE_UNKNOWN_WORD"] and i >= HYPERPARAMETERS["MONOLINGUAL_VOCABULARY_SIZE"]:
#            break
#        (cnt, w) = string.split(l)
#        words.append(w)
#
#    v = common.idmap.IDmap(words, allow_unknown=HYPERPARAMETERS["INCLUDE_UNKNOWN_WORD"])
#    assert v.len == HYPERPARAMETERS["MONOLINGUAL_VOCABULARY_SIZE"]
#    vocabulary.write(v)

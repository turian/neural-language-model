#!/usr/bin/env python
"""
Read in the w2w corpora (bi + monolingual), and build the translation
vocabulary (for each source word, what target words it can translate to).
Note: Each corpus is weighted in proportion to its length. (i.e. all
words are equally weighted.)
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
    import w2w.vocabulary
    import string
    from collections import defaultdict
    from common.mydict import sort as dictsort

    cnt = defaultdict(int)
    for l1, l2, f1, f2, falign in w2w.corpora.bicorpora_filenames():
        for (s1, s2, salign) in zip(open(f1), open(f2), open(falign)):
            ws1 = string.split(s1)
            ws2 = string.split(s2)
            for link in string.split(salign):
                i1, i2 = string.split(link, sep="-")
                cnt[l1, l2, ws1[int(i2)], ws2[int(i1)]] += 1

    for s in dictsort(cnt):
        print s
#    from collections import defaultdict
#    wordfreq = defaultdict(int)
#    for l1, l2, f1, f2, falign in w2w.corpora.bicorpora_filenames():
#        for w in readwords(f1): wordfreq[(l1,w)] += 1
#        for w in readwords(f2): wordfreq[(l2,w)] += 1
#
#    for l, f in w2w.corpora.monocorpora_filenames():
#        assert 0
#
#    words = {}
#    for (l, w) in wordfreq:
#        if l not in words: words[l] = []
#        if wordfreq[(l, w)] >= HYPERPARAMETERS["W2W MINIMUM WORD FREQUENCY"]:
#            words[l].append(w)
#
#    import common.idmap
#
#    for l in words:
#        v = common.idmap.IDmap(words[l], allow_unknown=HYPERPARAMETERS["INCLUDE_UNKNOWN_WORD"])
#        w2w.vocabulary.write(v, l)

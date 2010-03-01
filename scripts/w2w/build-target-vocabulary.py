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

    cnt = {}
    for l1, l2, f1, f2, falign in w2w.corpora.bicorpora_filenames():
        for (s1, s2, salign) in zip(open(f1), open(f2), open(falign)):
            # Read the two sentences and convert them to IDs.
            ws1 = [w2w.vocabulary.wordmap.id((l1, w1)) for w1 in string.split(s1)]
            ws2 = [w2w.vocabulary.wordmap.id((l2, w2)) for w2 in string.split(s2)]
#            print ws2, [w2w.vocabulary.wordmap.str(w2) for w2 in ws2]
            for link in string.split(salign):
                i2, i1 = string.split(link, sep="-")    # NB The order of the link indices is switched
                i1, i2 = int(i1), int(i2)
                w1 = ws1[i1]
                w2 = ws2[i2]
                if w1 not in cnt: cnt[w1] = defaultdict(int)
                cnt[w1][w2] += 1

#    for w1 in cnt:
#        print w2w.vocabulary.wordmap.str(w1), [(n, w2w.vocabulary.wordmap.str(w2)) for n, w2 in dictsort(cnt[w1])]

#    words = {}
#    for (l, w) in wordfreq:
#        if l not in words: words[l] = []
#        if wordfreq[(l, w)] >= HYPERPARAMETERS["W2W MINIMUM WORD FREQUENCY"]:
#            words[l].append(w)

    import w2w.targetvocabulary
    w2w.targetvocabulary.write(cnt)

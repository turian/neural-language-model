#!/usr/bin/env python
"""
Read in the w2w corpora (bi + monolingual), and build the translation
vocabulary (for each source word, what target words it can translate to).
Note: Each corpus is weighted in proportion to its length. (i.e. all
words are equally weighted.)
"""

import sys

if __name__ == "__main__":
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    HYPERPARAMETERS, options, args, newkeystr = common.options.reparse(HYPERPARAMETERS)
    import hyperparameters

    import logging
    logging.basicConfig(level=logging.DEBUG)

    import w2w.corpora
    from w2w.vocabulary import wordmap, language, wordform
    from collections import defaultdict
    from common.mydict import sort as dictsort

    cnt = {}
    for l1, l2, f1, f2, falign in w2w.corpora.bicorpora_filenames():
        for ws1, ws2, links in w2w.corpora.bicorpus_sentences_and_alignments(l1, l2, f1, f2, falign):
            for i1, i2 in links:
                w1 = ws1[i1]
                w2 = ws2[i2]
                if w1 not in cnt: cnt[w1] = {}
#                print wordmap.str(w1)[1], wordmap.str(w2)[1]

                l2new = language(w2)

                assert HYPERPARAMETERS["W2W SKIP TRANSLATIONS TO UNKNOWN WORD"]
                # Skip translations to unknown words
                if wordform(w2) == "*UNKNOWN*": continue

                assert l2new == l2
                if l2 not in cnt[w1]: cnt[w1][l2] = defaultdict(int)
                cnt[w1][l2][w2] += 1

    for w1 in cnt:
        for l2 in cnt[w1]:
            print wordmap().str(w1), l2, [(n, wordmap().str(w2)) for n, w2 in dictsort(cnt[w1][l2])]

#    words = {}
#    for (l, w) in wordfreq:
#        if l not in words: words[l] = []
#        if wordfreq[(l, w)] >= HYPERPARAMETERS["W2W MINIMUM WORD FREQUENCY"]:
#            words[l].append(w)

    import w2w.targetvocabulary
    w2w.targetvocabulary.write(cnt)

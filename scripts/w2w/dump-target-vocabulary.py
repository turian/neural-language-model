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
    from common.str import percent

    from vocabulary import wordmap, wordform, language
    from targetvocabulary import targetmap

    for w1 in wordmap().all:
        w1 = wordmap().id(w1)
        # Actually, should assert W2W SKIP TRANSLATIONS FROM UNKNOWN WORD
        assert HYPERPARAMETERS["W2W SKIP TRANSLATIONS TO UNKNOWN WORD"]
        if language(w1) is None:
            print >> sys.stderr, "Skipping %s" % `wordmap().str(w1)`
            continue
        if w1 not in targetmap():
            print >> sys.stderr, "Skipping %s, not a source word in targetmap" % `wordmap().str(w1)`
            continue
        for l2 in targetmap()[w1]:
            totcnt = 0
            for cnt, w2 in dictsort(targetmap()[w1][l2]): totcnt += cnt
            print wordmap().str(w1), l2, [(percent(cnt, totcnt), wordform(w2)) for cnt, w2 in dictsort(targetmap()[w1][l2])]

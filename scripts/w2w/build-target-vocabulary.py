#!/usr/bin/env python
"""
Read in the w2w corpora (bi + monolingual), and build the translation
vocabulary (for each source word, what target words it can translate to).
Note: Each corpus is weighted in proportion to its length. (i.e. all
words are equally weighted.)
"""

import sys
import itertools

if __name__ == "__main__":
    from common.stats import stats
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    HYPERPARAMETERS, options, args, newkeystr = common.options.reparse(HYPERPARAMETERS)
    import hyperparameters

    from common.str import percent

    import w2w.corpora
    import w2w.vocabulary
    import string
    from collections import defaultdict
    from common.mydict import sort as dictsort

    print >> sys.stderr, stats()

    cnt = {}
    for l1, l2, f1, f2, falign in w2w.corpora.bicorpora_filenames():
        i = 0
        emptycnt = 0
        print >> sys.stderr, "\n"
        print >> sys.stderr, "Processing %s, %s, %s, %s, %s" % (l1, l2, f1, f2, falign)
        fil1, fil2, filalign = open(f1), open(f2), open(falign)
        for (s1, s2, salign) in itertools.izip(fil1, fil2, filalign):
#            print s1, s2, salign,
            i += 1
            if i % 100000 == 0:
                print >> sys.stderr, "\tRead line %d of %s, %s, %s..." % (i, f1, f2, falign)
                print >> sys.stderr, "\tEmpty sentences are %s..." % (percent(emptycnt, i))
                print >> sys.stderr, "\t%s" % stats()
            # Read the two sentences and convert them to IDs.
            ws1 = [w2w.vocabulary.wordmap.id((l1, w1)) for w1 in string.split(s1)]
            ws2 = [w2w.vocabulary.wordmap.id((l2, w2)) for w2 in string.split(s2)]

            if len(ws1) == 0 or len(ws2) == 0:
                emptycnt += 1
                continue

#            print ws2, [w2w.vocabulary.wordmap.str(w2) for w2 in ws2]
            for link in string.split(salign):
                i1, i2 = string.split(link, sep="-")
                i1, i2 = int(i1), int(i2)
                w1 = ws1[i1]
                w2 = ws2[i2]
                if w1 not in cnt: cnt[w1] = defaultdict(int)
#                print w2w.vocabulary.wordmap.str(w1)[1], w2w.vocabulary.wordmap.str(w2)[1]
                cnt[w1][w2] += 1

        # Make sure all iterators are exhausted
        alldone = 0
        try: value = fil1.next()
        except StopIteration: alldone += 1
        try: value = fil2.next()
        except StopIteration: alldone += 1
        try: value = filalign.next()
        except StopIteration: alldone += 1
        assert alldone == 3

        print >> sys.stderr, "DONE. Read line %d of %s, %s, %s..." % (i, f1, f2, falign)
        print >> sys.stderr, "Empty sentences are %s..." % (percent(emptycnt, i))
        print >> sys.stderr, stats()

    for w1 in cnt:
        print w2w.vocabulary.wordmap.str(w1), [(n, w2w.vocabulary.wordmap.str(w2)) for n, w2 in dictsort(cnt[w1])]

#    words = {}
#    for (l, w) in wordfreq:
#        if l not in words: words[l] = []
#        if wordfreq[(l, w)] >= HYPERPARAMETERS["W2W MINIMUM WORD FREQUENCY"]:
#            words[l].append(w)

    import w2w.targetvocabulary
    w2w.targetvocabulary.write(cnt)

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
    reversecnt = {}
    for l1, l2, f1, f2, falign in w2w.corpora.bicorpora_filenames():
        for ws1, ws2, links in w2w.corpora.bicorpus_sentences_and_alignments(l1, l2, f1, f2, falign):
            for i1, i2 in links:
                if len(ws1) <= i1 or len(ws2) <= i2:
                    print >> sys.stderr, "This is going to break on link (%d, %d) because lens = (%d, %d)" % (i1,i2, len(ws1), len(ws2))
                    print >> sys.stderr, [wordform(w) for w in ws1]
                    print >> sys.stderr, [wordform(w) for w in ws2]
                    print >> sys.stderr, links
                w1 = ws1[i1]
                w2 = ws2[i2]
#                print wordmap.str(w1)[1], wordmap.str(w2)[1]

                l2new = language(w2)

                assert HYPERPARAMETERS["W2W SKIP TRANSLATIONS TO UNKNOWN WORD"]
                # Skip translations to unknown words
                if wordform(w2) == "*UNKNOWN*": continue

                assert l2new == l2


                # If we are filtering examples by lemma
                if not(HYPERPARAMETERS["W2W FOCUS LEMMAS"] is None or len (HYPERPARAMETERS["W2W FOCUS LEMMAS"]) == 0):
                    assert language(w1) == "en"
                    from lemmatizer import lemmatize
                    if lemmatize(language(w1), wordform(w1)) not in HYPERPARAMETERS["W2W FOCUS LEMMAS"]:
#                        logging.debug("Focus word %s (lemma %s) not in our list of focus lemmas" % (`wordmap().str(w1)`, lemmatize(language(w1), wordform(w1))))
                        continue

                if w1 not in cnt: cnt[w1] = {}
                if l2 not in cnt[w1]: cnt[w1][l2] = defaultdict(int)
                cnt[w1][l2][w2] += 1

                if w2 not in reversecnt: reversecnt[w2] = {}
                if l1 not in reversecnt[w2]: reversecnt[w2][l1] = defaultdict(int)
                reversecnt[w2][l1][w1] += 1

#    for w1 in cnt:
#        for l2 in cnt[w1]:
#            print wordmap().str(w1), l2, [(n, wordmap().str(w2)) for n, w2 in dictsort(cnt[w1][l2])]

#    words = {}
#    for (l, w) in wordfreq:
#        if l not in words: words[l] = []
#        if wordfreq[(l, w)] >= HYPERPARAMETERS["W2W MINIMUM WORD FREQUENCY"]:
#            words[l].append(w)

    import w2w.targetvocabulary
    w2w.targetvocabulary.write(cnt)
    w2w.targetvocabulary.write(reversecnt, name="reverse")

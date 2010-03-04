#!/usr/bin/env python
"""
Given embeddings in one language, initialize embeddings in all languages
using these monolingual embeddings.  We do this as a weighted average
of the translations of the target word in the embedding language.
(However, we only do the weighted average over words that have
embeddings. By comparison, we could do the weighted average and treat
words without embeddings as *UNKNOWN* in the embedding language, and
include these embeddings. But we don't.)
"""


if __name__ == "__main__":
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    HYPERPARAMETERS, options, args, newkeystr = common.options.reparse(HYPERPARAMETERS)
    import hyperparameters

    import sys
    from common.stats import stats
    from common.str import percent
    import common.file
    import numpy
    import string
    import copy

    import logging
    logging.basicConfig(level=logging.DEBUG)

    from w2w.vocabulary import wordmap, language, wordform
    from w2w.targetvocabulary import targetmap

    # Read in the embeddings
    print >> sys.stderr, "Reading embeddings from %s..." % HYPERPARAMETERS["W2W INITIAL EMBEDDINGS"]
    print >> sys.stderr, stats()
    original_embeddings = {}
    tot = 0
    for l in common.file.myopen(HYPERPARAMETERS["W2W INITIAL EMBEDDINGS"]):
        vals = string.split(l)
        word = vals[0]
        if HYPERPARAMETERS["W2W LOWERCASE INITIAL EMBEDDINGS BEFORE INITIALIZATION"] and word != "*UNKNOWN*":
            if (word[0] == '*' and word[-1] == '*' and len(word) > 1):
                print >> sys.stderr, "WEIRD WORD: %s" % word
            word = string.lower(word)
        assert len(vals[1:]) == HYPERPARAMETERS["EMBEDDING_SIZE"]
        tot += 1
        if word in original_embeddings:
#            print >> sys.stderr, "Skipping word %s (originally %s), we already have an embedding for it" % (word, vals[0])
            continue
        else:
            original_embeddings[word] = numpy.array([float(v) for v in vals[1:]])
    print >> sys.stderr, "...done reading embeddings from %s" % HYPERPARAMETERS["W2W INITIAL EMBEDDINGS"]
    print >> sys.stderr, "Skipped %s words for which we had duplicate embeddings" % percent(tot-len(original_embeddings), tot)
    print >> sys.stderr, stats()

    reversemap = targetmap(name="reverse")

    embeddings = numpy.zeros((wordmap().len, HYPERPARAMETERS["EMBEDDING_SIZE"]))
    assert embeddings.shape == (wordmap().len, HYPERPARAMETERS["EMBEDDING_SIZE"])

    ELANG = HYPERPARAMETERS["W2W INITIAL EMBEDDINGS LANGUAGE"]
    for w in range(wordmap().len):
        embedding = None
        # If this word is in a different language than the embeddings.
        if language(w) != HYPERPARAMETERS["W2W INITIAL EMBEDDINGS LANGUAGE"]:
            if w not in reversemap:
                print >> sys.stderr, "Word %s is not even in target map! Using *UNKNOWN*" % `wordmap().str(w)`
                embedding = original_embeddings["*UNKNOWN*"]
            elif ELANG not in reversemap[w]:
                print >> sys.stderr, "Have no %s translations for word %s, only have %s, using *UNKNOWN*" % (ELANG, wordmap().str(w), reversemap[w].keys())
                embedding = original_embeddings["*UNKNOWN*"]
            else:
                # Mix the target word embedding over the weighted translation into the source language

                mixcnt = {}
                for w2 in reversemap[w][ELANG]:
                    if language(w2) is None:
                        assert HYPERPARAMETERS["W2W SKIP TRANSLATIONS TO UNKNOWN WORD"]
                        continue
                    assert language(w2) == ELANG
                    if wordform(w2) not in original_embeddings:
                        print >> sys.stderr, "%s is NOT mixed by %s %d (no embedding)" % (wordmap().str(w), wordmap().str(w2), reversemap[w][ELANG][w2])
                        continue
                    mixcnt[w2] = reversemap[w][ELANG][w2]

                tot = 0
                for w2 in mixcnt: tot += mixcnt[w2]

                if tot == 0:
                    print >> sys.stderr, "Unable to mix ANY translations for %s, using *UNKNOWN*" % `wordmap().str(w)`
                    embedding = original_embeddings["*UNKNOWN*"]
                else:
                    embedding = numpy.zeros((HYPERPARAMETERS["EMBEDDING_SIZE"]))
                    for w2 in mixcnt:
                        embedding += 1. * mixcnt[w2] / tot * (original_embeddings[wordform(w2)])
#                       print >> sys.stderr, "%s is mixed %s by %s" % (wordmap().str(w), percent(mixcnt[w2], tot), wordmap().str(w2))
        else:
            if wordform(w) not in original_embeddings:
                print >> sys.stderr, "Word %s has no embedding, using *UNKNOWN*" % `wordmap().str(w)`
                embedding = original_embeddings["*UNKNOWN*"]
            else:
                embedding = original_embeddings[wordform(w)]
        embeddings[w] = copy.copy(embedding)

        print wordform(w), language(w),
        for v in embeddings[w]:
            print v,
        print

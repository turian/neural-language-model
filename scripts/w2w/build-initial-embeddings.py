#!/usr/bin/env python
"""
Read in W2W INITIAL EMBEDDINGS and construct initial embeddings over the vocabulary.
"""

## Use these embeddings to initialize the model
#W2W INITIAL EMBEDDINGS: /u/turian/data/share/embeddings-ACL2010-20100116-redo-baseline-with-100dims/model-2270000000.LEARNING_RATE=1e-09.EMBEDDING_LEARNING_RATE=1e-06.txt.gz
## Language of the initial embeddings
#W2W INITIAL EMBEDDINGS LANGUAGE: en
## Were the initial embeddings induced case-sensitive, but now we want to lowercase them?
#W2W LOWERCASE INITIAL EMBEDDINGS BEFORE INITIALIZATION: True


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

    import logging
    logging.basicConfig(level=logging.DEBUG)

    from w2w.vocabulary import wordmap, language, wordform

    embeddings = numpy.zeros((wordmap().len, HYPERPARAMETERS["EMBEDDING_SIZE"]))
    assert embeddings.shape == (wordmap().len, HYPERPARAMETERS["EMBEDDING_SIZE"])

    unknown_embeddings = None

    # Read in the embeddings
    print >> sys.stderr, "Reading embeddings from %s..." % HYPERPARAMETERS["W2W INITIAL EMBEDDINGS"]
    print >> sys.stderr, stats()
    original_embeddings = {}
    tot = 0
    for l in common.file.myopen(HYPERPARAMETERS["W2W INITIAL EMBEDDINGS"]):
        vals = string.split(l)
        word = vals[0]
        if HYPERPARAMETERS["W2W LOWERCASE INITIAL EMBEDDINGS BEFORE INITIALIZATION"]:
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

#    for w in range(wordmap().len):
#        print wordform(w), language(w)

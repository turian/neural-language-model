"""
Streaming examples.
"""

from w2w.corpora import bicorpora_filenames, monocorpora_filenames, bicorpus_sentences_and_alignments
from common.file import myopen

from w2w.targetvocabulary import targetmap
from w2w.vocabulary import wordmap, language, wordform
import string
import logging

def get_training_biexample(l1, l2, f1, f2, falign):
    """
    Generator of bilingual training examples from this bicorpus.
    Each example is of the form:
        ((l1, seq), w2)
    where l1 is the source language, seq is a sequence of word ids in
    the source language, and w2 is the word id of the focus word in the
    target language.
    """
    import common.hyperparameters
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    WINDOW = HYPERPARAMETERS["WINDOW_SIZE"]

    for ws1, ws2, links in bicorpus_sentences_and_alignments(l1, l2, f1, f2, falign):
        for i1, i2 in links:
            w1 = ws1[i1]
            w2 = ws2[i2]

            l2new = language(w2)
            assert HYPERPARAMETERS["W2W SKIP TRANSLATIONS TO UNKNOWN WORD"]
            # Skip translations to unknown words
            if wordform(w2) == "*UNKNOWN*": continue
            assert l2new == l2

            if w1 not in targetmap():
                logging.warning("No translations for word %s, skipping" % (`wordmap().str(w1)`))
                continue

            if l2new not in targetmap()[w1]:
                logging.warning("Word %s has no translations for language %s, skipping" % (`wordmap().str(w1)`, l2new))
                continue

            if w2 not in targetmap()[w1][l2new]:
                logging.error("Word %s cannot translate to word %s, skipping" % (`wordmap().str(w1)`, `wordmap().str(w2)`))
                continue

            if len(targetmap()[w1][l2new]) == 1:
                logging.info("Word %s has only one translation in language %s, skipping" % (`wordmap().str(w1)`, l2new))
                continue

            # Extract the window of tokens around index i1. Pad with *LBOUNDARY* and *RBOUNDARY* as necessary.
            min = i1 - (WINDOW-1)/2
            max = i1 + (WINDOW-1)/2
            lpad = 0
            rpad = 0
            if min < 0:
                lpad = -min
                min = 0
            if max >= len(ws1):
                rpad = max - (len(ws1)-1)
                max = len(ws1)-1
            assert lpad + (max - min + 1) + rpad == WINDOW

#            print i1 - (WINDOW-1)/2, i1 + (WINDOW-1)/2
#            print "min=%d, max=%d, lpad=%d, rpad=%d" % (min, max, lpad, rpad)
            seq = [wordmap().id((None, "*LBOUNDARY*"))]*lpad + ws1[min:max+1] + [wordmap().id((None, "*RBOUNDARY*"))]*rpad
#            print [wordmap.str(w) for w in seq]
            assert len(seq) == WINDOW
#            print ws1[i1 - (WINDOW-1)/2:i1 + (WINDOW-1)/2]

            yield (l1, seq), w2

def get_training_minibatch():
    import common.hyperparameters
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    MINIBATCH_SIZE = HYPERPARAMETERS["MINIBATCH SIZE"]

    generators = []
    for l1, l2, f1, f2, falign in bicorpora_filenames():
#        print l1, l2, f1, f2, falign
        generators.append(get_training_biexample(l1, l2, f1, f2, falign))
    for l, f in monocorpora_filenames(): assert 0

    # Cycles over generators.
    idx = 0
    last_minibatch = None
    while 1:
        minibatch = []
        for e in generators[idx]:
            minibatch.append(e)
            if len(minibatch) >= MINIBATCH_SIZE:
                break
        if len(minibatch) > 0:
            last_minibatch = idx
            yield minibatch
        elif last_minibatch == idx:
            # We haven't had any minibatch in the last cycle over the generators.
            # So we are done will all corpora.
            break

        # Go to the next corpus
        idx = (idx + 1) % len(generators)

if __name__ == "__main__":
    for minibatch in get_training_minibatch():
#        print len(minibatch)
        for (l1, seq), w2 in minibatch:
            print (l1, [wordmap().str(w)[1] for w in seq]), wordmap().str(w2)

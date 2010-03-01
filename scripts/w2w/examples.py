"""
Streaming examples.
"""

from w2w.corpora import bicorpora_filenames, monocorpora_filenames
from common.file import myopen

from w2w.targetvocabulary import targetmap
from w2w.vocabulary import wordmap
import string

def get_training_biexample(l1, l2, f1, f2, falign):
    """
    Generator of bilingual training examples from this bicorpus.
    """
    import common.hyperparameters
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    WINDOW = HYPERPARAMETERS["WINDOW_SIZE"]

    assert 0        # Share code with  w2w/build-target-vocabulary.py
    for (s1, s2, salign) in zip(open(f1), open(f2), open(falign)):
        # Read the two sentences and convert them to IDs.
        ws1 = [wordmap.id((l1, w1)) for w1 in string.split(s1)]
        ws2 = [wordmap.id((l2, w2)) for w2 in string.split(s2)]
        for link in string.split(salign):
            i1, i2 = string.split(link, sep="-")
            i1, i2 = int(i1), int(i2)
            w1 = ws1[i1]
            w2 = ws2[i2]
            if w1 not in targetmap or w2 not in targetmap[w1]:
                print >> sys.stderr, "Translating %s to %s is not in target map, skipping" % (wordmap.str(w1), wordmap.str(w2))
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
            seq = [wordmap.id((None, "*LBOUNDARY*"))]*lpad + ws1[min:max+1] + [wordmap.id((None, "*RBOUNDARY*"))]*rpad
#            print [wordmap.str(w) for w in seq]
            assert len(seq) == WINDOW
#            print ws1[i1 - (WINDOW-1)/2:i1 + (WINDOW-1)/2]
            yield seq, w2

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
        for seq, w2 in minibatch:
            print [wordmap.str(w)[1] for w in seq], wordmap.str(w2)

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
    for (s1, s2, salign) in zip(open(f1), open(f2), open(falign)):
        # Read the two sentences and convert them to IDs.
        ws1 = [wordmap.id((l1, w1)) for w1 in string.split(s1)]
        ws2 = [wordmap.id((l2, w2)) for w2 in string.split(s2)]
        for link in string.split(salign):
            i1, i2 = string.split(link, sep="-")
            w1 = ws1[int(i2)]
            w2 = ws2[int(i1)]
            if w1 not in targetmap or w2 not in targetmap[w1]:
                print >> sys.stderr, "Translating %s to %s is not in target map, skipping" % (wordmap.str(w1), wordmap.str(w2))
                continue
            yield w1, w2

def get_training_minibatch():
    generators = []
    for l1, l2, f1, f2, falign in bicorpora_filenames():
        generators.append(get_training_biexample(l1, l2, f1, f2, falign))
    for l, f in monocorpora_filenames(): assert 0

    # FIXME: This cycles over generators in the OUTER loop. Want it in the inner loop!
    for g in generators:
        for e in g:
            yield e

if __name__ == "__main__":
    for w1, w2 in get_training_minibatch():
        print wordmap.str(w1), wordmap.str(w2)

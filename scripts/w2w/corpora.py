"""
Methods for reading corpora.
"""

from os.path import join, isdir, exists
import sys
import os
import re
import itertools
import string

from common.stats import stats
from common.str import percent

def bicorpora_filenames():
    """
    For each bicorpora language pair in "W2W BICORPORA", traverse that
    language pair's subdirectory of DATA_DIR. Find all corpora files in
    that directory.
    Generator yields: tuples of type (l1, l2, f1, f2, falign), where l1 =
    source language, l2 = target language, f1 = source filename, f2 =
    target filename, falign = alignment file.
    """
    import common.hyperparameters, hyperparameters
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    
    for (l1, l2) in HYPERPARAMETERS["W2W BICORPORA"]:
        d = join(HYPERPARAMETERS["DATA_DIR"], "%s-%s" % (l1, l2))
        assert isdir(d)
        l1re = re.compile("%s$" % l1)
        alignre = re.compile("align.*-%s$" % l1)
        for f1 in os.listdir(d):
            f1 = join(d, f1)
            if not l1re.search(f1) or alignre.search(f1): continue
            f2 = l1re.sub(l2, f1)
            assert exists(f2)
            falign = l1re.sub("align.%s-%s" % (l1, l2), f1)
            assert exists(falign)
            yield l1, l2, f1, f2, falign

def monocorpora_filenames():
    import common.hyperparameters, hyperparameters
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    # Not yet implemented
    assert len(HYPERPARAMETERS["W2W MONOCORPORA"]) == 0
    return []

def bicorpus_sentences_and_alignments(l1, l2, f1, f2, falign):
    """
    Given languages l1 and l2 and their bicorpus filenames f1, f2, and falign,
    yield tuples of the former (ws1, ws2, links),
    where ws1 are the word ids in the sentence from f1,
    where ws2 are the word ids in the sentence from f2,
    and links is a list of (i1, i2) word indexes that are linked.
    """
    from w2w.vocabulary import wordmap

    i = 0
    emptycnt = 0
    print >> sys.stderr, "\n"
    print >> sys.stderr, "Reading %s,%s sentences and alignments from %s, %s, %s" % (l1, l2, f1, f2, falign)
    fil1, fil2, filalign = open(f1), open(f2), open(falign)
    for (s1, s2, salign) in itertools.izip(fil1, fil2, filalign):
   #     print s1, s2, salign,
        i += 1
        if i % 100000 == 0:
            print >> sys.stderr, "\tRead line %d of %s, %s, %s..." % (i, f1, f2, falign)
            print >> sys.stderr, "\tEmpty sentences are %s..." % (percent(emptycnt, i))
            print >> sys.stderr, "\t%s" % stats()

        ws1 = [(l1, w1) for w1 in string.split(s1)]
        ws2 = [(l2, w2) for w2 in string.split(s2)]
        ws1 = [wordmap.id(tok) for tok in ws1]
        ws2 = [wordmap.id(tok) for tok in ws2]
   
        if len(ws1) == 0 or len(ws2) == 0:
            emptycnt += 1
            continue
   
   #     print ws2, [w2w.vocabulary.wordmap.str(w2) for w2 in ws2]
        links = [string.split(link, sep="-") for link in string.split(salign)]
        links = [(int(i1), int(i2)) for i1, i2 in links]

        yield ws1, ws2, links
   
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

if __name__ == "__main__":
    for l1, l2, f1, f2, falign in bicorpora_filenames():
        print l1, l2, f1, f2, falign
    print monocorpora_filenames()

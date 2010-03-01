"""
Methods for reading corpora.
"""

from os.path import join, isdir, exists
import os
import re

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

if __name__ == "__main__":
    for l1, l2, f1, f2, falign in bicorpora_filenames():
        print l1, l2, f1, f2, falign
    print monocorpora_filenames()

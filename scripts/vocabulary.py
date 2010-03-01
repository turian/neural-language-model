"""
Automatically load the wordmap, if available.
"""

import cPickle
from common.file import myopen
import sys

def _wordmap_filename(name):
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    return HYPERPARAMETERS["MONOLINGUAL_VOCABULARY_IDMAP_FILE"]

wordmap = None
try:
    wordmap = cPickle.load(myopen(_wordmap_filename()))
    wordmap.str = wordmap.key
except: pass

def write(wordmap, name=""):
    """
    Write the word ID map, passed as a parameter.
    """
    print >> sys.stderr, "Writing word map to %s..." % _wordmap_filename(name)
    cPickle.dump(wordmap, myopen(_wordmap_filename(name), "w"))

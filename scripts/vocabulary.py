"""
Automatically load the wordmap, if available.
"""

import hyperparameters
import cPickle
from common.file import myopen
import sys

def _wordmap_filename():
    return hyperparameters.VOCABULARY_IDMAP_FILE

wordmap = None
try:
    wordmap = cPickle.load(myopen(_wordmap_filename()))
except: pass

def write(wordmap):
    """
    Write the word ID map, passed as a parameter.
    """
    print >> sys.stderr, "Writing word map to %s..." % _wordmap_filename()
    cPickle.dump(wordmap, myopen(_wordmap_filename(), "w"))

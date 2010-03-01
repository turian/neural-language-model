"""
Automatically load the wordmap, if available.
"""

import cPickle
from common.file import myopen
import sys
from os.path import join

def _wordmap_filename():
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    return join(HYPERPARAMETERS["DATA_DIR"], "idmap.minfreq=%d.include_unknown=%s.pkl.gz" % (HYPERPARAMETERS["W2W MINIMUM WORD FREQUENCY"], HYPERPARAMETERS["INCLUDE_UNKNOWN_WORD"]))

wordmap = None
try:
    wordmap = cPickle.load(myopen(_wordmap_filename()))
    wordmap.str = wordmap.key
except: pass

def write(_wordmap):
    """
    Write the word ID map, passed as a parameter.
    """
    global wordmap
    wordmap = _wordmap
    print >> sys.stderr, "Writing word map with %d words to %s..." % (_wordmap.len, _wordmap_filename())
    cPickle.dump(wordmap, myopen(_wordmap_filename(), "w"))

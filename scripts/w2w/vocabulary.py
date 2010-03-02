"""
Automatically load the wordmap, if available.
"""

import cPickle
from common.file import myopen
import sys
from os.path import join

def _wordmap_filename():
    import common.hyperparameters, common.options, hyperparameters
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    return join(HYPERPARAMETERS["DATA_DIR"], "idmap.minfreq=%d.include_unknown=%s.pkl.gz" % (HYPERPARAMETERS["W2W MINIMUM WORD FREQUENCY"], HYPERPARAMETERS["INCLUDE_UNKNOWN_WORD"]))

_wordmap = None
def wordmap():
    global _wordmap
    if _wordmap is None:
        _wordmap = cPickle.load(myopen(_wordmap_filename()))
        _wordmap.str = _wordmap.key
    return _wordmap

def write(_wordmap_new):
    """
    Write the word ID map, passed as a parameter.
    """
    global _wordmap
    assert _wordmap is None
    _wordmap = _wordmap_new
    print >> sys.stderr, "Writing word map with %d words to %s..." % (_wordmap.len, _wordmap_filename())
    cPickle.dump(_wordmap, myopen(_wordmap_filename(), "w"))

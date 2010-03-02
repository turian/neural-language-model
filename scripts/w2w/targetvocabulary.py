"""
targetmap[w1][l2][w2] = c means that source word ID w1 mapped to target
language l2 and target word ID w2 with count c.
"""

import cPickle
from common.file import myopen
import sys
from os.path import join

def _targetmap_filename():
    import common.hyperparameters, common.options, hyperparameters
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    return join(HYPERPARAMETERS["DATA_DIR"], "targetmap.minfreq=%d.include_unknown=%s.pkl.gz" % (HYPERPARAMETERS["W2W MINIMUM WORD FREQUENCY"], HYPERPARAMETERS["INCLUDE_UNKNOWN_WORD"]))

_targetmap = None
def targetmap():
    global _targetmap
    if _targetmap is None:
        _targetmap = cPickle.load(myopen(_targetmap_filename()))
    return _targetmap

def write(_targetmap_new):
    """
    Write the word ID map, passed as a parameter.
    """
    global _targetmap
    assert _targetmap is None
    _targetmap = _targetmap_new
    print >> sys.stderr, "Writing target map to %s..." % _targetmap_filename()
    cPickle.dump(_targetmap, myopen(_targetmap_filename(), "w"))

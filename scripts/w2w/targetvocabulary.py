"""
targetmap[w1][l2][w2] = c means that source word ID w1 mapped to target
language l2 and target word ID w2 with count c.
"""

import cPickle
from common.file import myopen
import sys
from os.path import join

def _targetmap_filename(name=""):
    import common.hyperparameters, common.options, hyperparameters
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    return join(HYPERPARAMETERS["DATA_DIR"], "%stargetmap.minfreq=%d.include_unknown=%s.pkl.gz" % (name, HYPERPARAMETERS["W2W MINIMUM WORD FREQUENCY"], HYPERPARAMETERS["INCLUDE_UNKNOWN_WORD"]))

_targetmap = {}
def targetmap(name=""):
    global _targetmap
    if name not in _targetmap:
        _targetmap[name] = cPickle.load(myopen(_targetmap_filename(name=name)))
    return _targetmap[name]

def write(_targetmap_new, name=""):
    """
    Write the word ID map, passed as a parameter.
    """
    global _targetmap
    assert name not in _targetmap
    _targetmap[name] = _targetmap_new
    f = _targetmap_filename(name=name)
    print >> sys.stderr, "Writing target map to %s..." % f
    cPickle.dump(_targetmap[name], myopen(f, "w"))

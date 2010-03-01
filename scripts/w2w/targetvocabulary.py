"""
targetmap[w1][w2] = c means that source word ID w1 mapped to target word ID w2 with count c.
Automatically load the targetmap, if available.
"""

import cPickle
from common.file import myopen
import sys
from os.path import join

def _targetmap_filename():
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    return join(HYPERPARAMETERS["DATA_DIR"], "targetmap.minfreq=%d.include_unknown=%s.pkl.gz" % (HYPERPARAMETERS["W2W MINIMUM WORD FREQUENCY"], HYPERPARAMETERS["INCLUDE_UNKNOWN_WORD"]))

targetmap = None
try:
    targetmap = cPickle.load(myopen(_targetmap_filename()))
except: pass

def write(_targetmap):
    """
    Write the word ID map, passed as a parameter.
    """
    global targetmap
    targetmap = _targetmap
    print >> sys.stderr, "Writing target map to %s..." % _targetmap_filename()
    cPickle.dump(targetmap, myopen(_targetmap_filename(), "w"))

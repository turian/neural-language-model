"""
Save and load training state.
@todo: Training state variables (cnt, epoch) should all be combined into one object.
"""

import logging
import os.path
import cPickle

from common.stats import stats
from common.file import myopen
import common.json
import sys

_lastfilename = None
def save(translation_model, cnt, lastcnt, epoch, rundir, newkeystr):
    global _lastfilename

    filename = os.path.join(rundir, "translation_model-%d%s.pkl" % (cnt, newkeystr))
    logging.info("Writing translation_model to %s..." % filename)
    logging.info(stats())
    cPickle.dump(translation_model, myopen(filename, "wb"), protocol=-1)
    logging.info("...done writing translation_model to %s" % filename)
    logging.info(stats())

#    if _lastfilename is not None:
#        logging.info("Removing old translation_model %s..." % _lastfilename)
#        try:
#            os.remove(_lastfilename)
#            logging.info("...removed %s" % _lastfilename)
#        except:
#            logging.info("Could NOT remove %s" % _lastfilename)
    _lastfilename = filename

    common.json.dumpfile((cnt, lastcnt, epoch, filename), os.path.join(rundir, "trainstate.json"))

    filename = os.path.join(rundir, "newkeystr.txt")
    myopen(filename, "wt").write(newkeystr)

def load(rundir, newkeystr):
    """
    Read the directory and load the translation_model, the training count, the training epoch, and the training state.
    """
    global _lastfilename

    filename = os.path.join(rundir, "newkeystr.txt")
    assert newkeystr == myopen(filename).read()

    (cnt, lastcnt, epoch, filename) = common.json.loadfile(os.path.join(rundir, "trainstate.json"))

#    filename = os.path.join(rundir, "translation_model-%d%s.pkl" % (cnt, newkeystr))
    print >> sys.stderr, ("Reading translation_model from %s..." % filename)
    print >> sys.stderr, (stats())
    translation_model = cPickle.load(myopen(filename))
    print >> sys.stderr, ("...done reading translation_model from %s" % filename)
    print >> sys.stderr, (stats())
    _lastfilename = filename

    return (translation_model, cnt, lastcnt, epoch)

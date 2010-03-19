"""
Save and load training state.
@todo: Training state variables (cnt, epoch, trainstate) should all be combined into one object.
"""

import logging
import os.path
import cPickle

from common.stats import stats
from common.file import myopen
import sys

_lastfilename = None
def save(model, cnt, epoch, trainstate, rundir, newkeystr):
    global _lastfilename

    filename = os.path.join(rundir, "model-%d%s.pkl" % (cnt, newkeystr))
    logging.info("Writing model to %s..." % filename)
    logging.info(stats())
    cPickle.dump(model, myopen(filename, "wb"), protocol=-1)
    logging.info("...done writing model to %s" % filename)
    logging.info(stats())

    if _lastfilename is not None:
        logging.info("Removing old model %s..." % _lastfilename)
        try:
            os.remove(_lastfilename)
            logging.info("...removed %s" % _lastfilename)
        except:
            logging.info("Could NOT remove %s" % _lastfilename)
    _lastfilename = filename

    filename = os.path.join(rundir, "trainstate.pkl")
    cPickle.dump((trainstate, cnt, epoch), myopen(filename, "wb"), protocol=-1)

    filename = os.path.join(rundir, "newkeystr.txt")
    myopen(filename, "wt").write(newkeystr)

def load(rundir, newkeystr):
    """
    Read the directory and load the model, the training count, the training epoch, and the training state.
    """
    global _lastfilename

    filename = os.path.join(rundir, "newkeystr.txt")
    assert newkeystr == myopen(filename).read()

    filename = os.path.join(rundir, "trainstate.pkl")
    (trainstate, cnt, epoch) = cPickle.load(myopen(filename))

    filename = os.path.join(rundir, "model-%d%s.pkl" % (cnt, newkeystr))
    print >> sys.stderr, ("Reading model from %s..." % filename)
    print >> sys.stderr, (stats())
    model = cPickle.load(myopen(filename))
    print >> sys.stderr, ("...done reading model from %s" % filename)
    print >> sys.stderr, (stats())
    _lastfilename = filename

    return (model, cnt, epoch, trainstate)

"""
Methods for getting examples.
"""

from common.stats import stats
from common.file import myopen
import string

import common.hyperparameters
import sys

class TrainingExampleStream(object):
    def __init__(self):
        self.count = 0
        pass
    
    def __iter__(self):
        HYPERPARAMETERS = common.hyperparameters.read("language-model")
        from vocabulary import wordmap
        self.filename = HYPERPARAMETERS["TRAIN_SENTENCES"]
        self.count = 0
        for l in myopen(self.filename):
            prevwords = []
            for w in string.split(l):
                w = string.strip(w)
                id = None
                if wordmap.exists(w):
                    prevwords.append(wordmap.id(w))
                    if len(prevwords) >= HYPERPARAMETERS["WINDOW_SIZE"]:
                        self.count += 1
                        yield prevwords[-HYPERPARAMETERS["WINDOW_SIZE"]:]
                else:
                    prevwords = []

    def __getstate__(self):
        return self.filename, self.count

    def __setstate__(self, state):
        """
        @warning: We ignore the filename.  If we wanted
        to be really fastidious, we would assume that
        HYPERPARAMETERS["TRAIN_SENTENCES"] might change.  The only
        problem is that if we change filesystems, the filename
        might change just because the base file is in a different
        path. So we issue a warning if the filename is different from
        """
        filename, count = state
        print >> sys.stderr, ("__setstate__(%s)..." % `state`)
        print >> sys.stderr, (stats())
        iter = self.__iter__()
        while count != self.count:
#            print count, self.count
            iter.next()
        if self.filename != filename:
            assert self.filename == HYPERPARAMETERS["TRAIN_SENTENCES"]
            print >> sys.stderr, ("self.filename %s != filename given to __setstate__ %s" % (self.filename, filename))
        print >> sys.stderr, ("...__setstate__(%s)" % `state`)
        print >> sys.stderr, (stats())

class TrainingMinibatchStream(object):
    def __init__(self):
        pass
    
    def __iter__(self):
        HYPERPARAMETERS = common.hyperparameters.read("language-model")
        minibatch = []
        self.get_train_example = TrainingExampleStream()
        for e in self.get_train_example:
#            print self.get_train_example.__getstate__()
            minibatch.append(e)
            if len(minibatch) >= HYPERPARAMETERS["MINIBATCH SIZE"]:
                assert len(minibatch) == HYPERPARAMETERS["MINIBATCH SIZE"]
                yield minibatch
                minibatch = []

    def __getstate__(self):
        return (self.get_train_example.__getstate__(),)

    def __setstate__(self, state):
        """
        @warning: We ignore the filename.
        """
        self.get_train_example = TrainingExampleStream()
        self.get_train_example.__setstate__(state[0])

def get_validation_example():
    HYPERPARAMETERS = common.hyperparameters.read("language-model")

    from vocabulary import wordmap
    for l in myopen(HYPERPARAMETERS["VALIDATION_SENTENCES"]):
        prevwords = []
        for w in string.split(l):
            w = string.strip(w)
            if wordmap.exists(w):
                prevwords.append(wordmap.id(w))
                if len(prevwords) >= HYPERPARAMETERS["WINDOW_SIZE"]:
                    yield prevwords[-HYPERPARAMETERS["WINDOW_SIZE"]:]
            else:
                prevwords = []

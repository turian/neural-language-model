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
                    # If we can learn an unknown word token, we should
                    # delexicalize the word, not discard the example!
                    if HYPERPARAMETERS["INCLUDE_UNKNOWN_WORD"]: assert 0
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
                # If we can learn an unknown word token, we should
                # delexicalize the word, not discard the example!
                if HYPERPARAMETERS["INCLUDE_UNKNOWN_WORD"]: assert 0
                prevwords = []

def corrupt_example(model, e):
    """
    Return a corrupted version of example e, plus the weight of this example.
    """
    from hyperparameters import HYPERPARAMETERS
    import random
    import copy
    e = copy.copy(e)
    last = e[-1]
    cnt = 0
    while e[-1] == last:
        if HYPERPARAMETERS["NGRAM_FOR_TRAINING_NOISE"] == 0:
            e[-1] = random.randint(0, model.parameters.vocab_size-1)
            pr = 1./model.parameters.vocab_size
        elif HYPERPARAMETERS["NGRAM_FOR_TRAINING_NOISE"] == 1:
            import noise
            from common.myrandom import weighted_sample
            e[-1], pr = weighted_sample(noise.indexed_weights())
#            from vocabulary import wordmap
#            print wordmap.str(e[-1]), pr
        else:
            assert 0
        cnt += 1
        # Backoff to 0gram smoothing if we fail 10 times to get noise.
        if cnt > 10: e[-1] = random.randint(0, model.parameters.vocab_size-1)
    weight = 1./pr
    return e, weight

def corrupt_examples(model, correct_sequences):
    noise_sequences = []
    weights = []
    for e in correct_sequences:
        noise_sequence, weight = model.corrupt_example(e)
        noise_sequences.append(noise_sequence)
        weights.append(weight)
    return noise_sequences, weights

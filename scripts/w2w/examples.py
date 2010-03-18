"""
Streaming examples.
"""

from w2w.corpora import bicorpora_filenames, monocorpora_filenames, bicorpus_sentences_and_alignments
from common.file import myopen
from common.stats import stats

from w2w.targetvocabulary import targetmap
from w2w.vocabulary import wordmap, language, wordform
import string
import logging

import random
from rundir import rundir
import os.path
import cPickle

class BilingualExample:
    def __init__(self, l1, l1seq, w1, w2):
        """
        l1 = source language
        l1seq = sequence of word IDs in source language
        w1 = focus word ID in source language
        w2 = focus word ID in target language
        """
        self.l1 = l1
        self.l1seq = l1seq
        self.w1 = w1
        self.w2 = w2

        if wordform(self.w1) != "*UNKNOWN*":
            assert self.l1 == language(self.w1)

    @property
    def l2(self):
        return language(self.w2)

    @property
    def corrupt(self):
        """
        Return a (notw2, weight), a corrupt target word and its weight.
        Note: This will return a different random value every call.
        """
        from hyperparameters import HYPERPARAMETERS
        import random
        possible_targets = targetmap()[self.w1][self.l2]
        assert len(possible_targets) > 1
        assert self.w2 in possible_targets
        notw2 = self.w2
        cnt = 0
        while self.w2 == notw2:
            if HYPERPARAMETERS["NGRAM_FOR_TRAINING_NOISE"] == 0:
                notw2 = random.choice(possible_targets)
                pr = 1./len(possible_targets)
            elif HYPERPARAMETERS["NGRAM_FOR_TRAINING_NOISE"] == 1:
                assert 0
    #            import noise
    #            from common.myrandom import weighted_sample
    #            e[-1], pr = weighted_sample(noise.indexed_weights())
    ##            from vocabulary import wordmap
    ##            print wordmap.str(e[-1]), pr
            else:
                assert 0
            cnt += 1
            # Backoff to 0gram smoothing if we fail 10 times to get noise.
            if cnt > 10: notw2 = random.choice(possible_targets)

        if HYPERPARAMETERS["UNIFORM EXAMPLE WEIGHTS"]:
            weight = 1.
        else:
            weight = 1./pr
        return notw2, weight

    def __str__(self):
        return "%s" % `(wordmap().str(self.w2), self.l1, wordform(self.w1), [wordmap().str(w)[1] for w in self.l1seq])`

def get_training_biexample(l1, l2, f1, f2, falign):
    """
    Generator of bilingual training examples from this bicorpus.
    Each example is of the form:
        ((l1, seq), w2)
    where l1 is the source language, seq is a sequence of word ids in
    the source language, and w2 is the word id of the focus word in the
    target language.
    """
    import common.hyperparameters
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    WINDOW = HYPERPARAMETERS["WINDOW_SIZE"]

    for ws1, ws2, links in bicorpus_sentences_and_alignments(l1, l2, f1, f2, falign):
        for i1, i2 in links:
            w1 = ws1[i1]
            w2 = ws2[i2]

            l2new = language(w2)
            assert HYPERPARAMETERS["W2W SKIP TRANSLATIONS TO UNKNOWN WORD"]
            # Skip translations to unknown words
            if wordform(w2) == "*UNKNOWN*": continue
            assert l2new == l2

            # Skip translations from unknown words
            if wordform(w1) == "*UNKNOWN*": continue

            # If we are filtering examples by lemma
            if not(HYPERPARAMETERS["W2W FOCUS LEMMAS"] is None or len (HYPERPARAMETERS["W2W FOCUS LEMMAS"]) == 0):
#                print wordmap().str(w1), wordmap().str(w2)
                assert language(w1) == "en"
                from lemmatizer import lemmatize
                if lemmatize(language(w1), wordform(w1)) not in HYPERPARAMETERS["W2W FOCUS LEMMAS"]:
                    logging.debug("Focus word %s (lemma %s) not in our list of focus lemmas" % (`wordmap().str(w1)`, lemmatize(language(w1), wordform(w1))))
                    continue

            if w1 not in targetmap():
                logging.warning("No translations for word %s, skipping" % (`wordmap().str(w1)`))
                continue

            if l2new not in targetmap()[w1]:
                logging.warning("Word %s has no translations for language %s, skipping" % (`wordmap().str(w1)`, l2new))
                continue

            if w2 not in targetmap()[w1][l2new]:
                logging.error("Word %s cannot translate to word %s, skipping" % (`wordmap().str(w1)`, `wordmap().str(w2)`))
                continue

            if len(targetmap()[w1][l2new]) == 1:
                logging.debug("Word %s has only one translation in language %s, skipping" % (`wordmap().str(w1)`, l2new))
                continue

            # Extract the window of tokens around index i1. Pad with *LBOUNDARY* and *RBOUNDARY* as necessary.
            min = i1 - (WINDOW-1)/2
            max = i1 + (WINDOW-1)/2
            lpad = 0
            rpad = 0
            if min < 0:
                lpad = -min
                min = 0
            if max >= len(ws1):
                rpad = max - (len(ws1)-1)
                max = len(ws1)-1
            assert lpad + (max - min + 1) + rpad == WINDOW

#            print i1 - (WINDOW-1)/2, i1 + (WINDOW-1)/2
#            print "min=%d, max=%d, lpad=%d, rpad=%d" % (min, max, lpad, rpad)
            seq = [wordmap().id((None, "*LBOUNDARY*"))]*lpad + ws1[min:max+1] + [wordmap().id((None, "*RBOUNDARY*"))]*rpad
#            print [wordmap.str(w) for w in seq]
            assert len(seq) == WINDOW
#            print ws1[i1 - (WINDOW-1)/2:i1 + (WINDOW-1)/2]

            assert seq[(WINDOW-1)/2] == w1
            yield BilingualExample(l1, seq, w1, w2)

def get_training_minibatch_online():
    """
    Warning: The approach has the weird property that if one language
    pair's corpus is way longer than others, it will be the only examples
    for a while after the other corpora are exhausted.
    """

    import common.hyperparameters
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    MINIBATCH_SIZE = HYPERPARAMETERS["MINIBATCH SIZE"]

    generators = []
    for l1, l2, f1, f2, falign in bicorpora_filenames():
#        print l1, l2, f1, f2, falign
        generators.append(get_training_biexample(l1, l2, f1, f2, falign))
    for l, f in monocorpora_filenames(): assert 0

    # Cycles over generators.
    idx = 0
    last_minibatch = None
    while 1:
        minibatch = []
        for e in generators[idx]:
            minibatch.append(e)
            if len(minibatch) >= MINIBATCH_SIZE:
                break
        if len(minibatch) > 0:
            last_minibatch = idx
            yield minibatch
        elif last_minibatch == idx:
            # We haven't had any minibatch in the last cycle over the generators.
            # So we are done will all corpora.
            break

        # Go to the next corpus
        idx = (idx + 1) % len(generators)

def training_examples_cache_filename():
    import common.hyperparameters, hyperparameters
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    return os.path.join(HYPERPARAMETERS["DATA_DIR"], "examples-cache.minfreq=%d.include_unknown=%s.window-%d.pkl.gz" % (HYPERPARAMETERS["W2W MINIMUM WORD FREQUENCY"], HYPERPARAMETERS["INCLUDE_UNKNOWN_WORD"], HYPERPARAMETERS["WINDOW_SIZE"]))

_all_examples = None
def all_training_examples_cached():
    global _all_examples
    if _all_examples is None:
        try:
            _all_examples, cnt = cPickle.load(myopen(training_examples_cache_filename()))
            assert len(_all_examples) == cnt
            logging.info("Successfully read %d training examples from %s" % (cnt, training_examples_cache_filename()))
            logging.info(stats())
        except:
            logging.info("(Couldn't read training examples from %s, sorry)" % (training_examples_cache_filename()))
            logging.info("Caching all training examples...")
            logging.info(stats())
            _all_examples = []
            for l1, l2, f1, f2, falign in bicorpora_filenames():
                for e in get_training_biexample(l1, l2, f1, f2, falign):
                    _all_examples.append(e)
                    if len(_all_examples) % 10000 == 0:
                        logging.info("\tcurrently have read %d training examples" % len(_all_examples))
                        logging.info(stats())
            random.shuffle(_all_examples)
            logging.info("...done caching all %d training examples" % len(_all_examples))
            logging.info(stats())

            cnt = len(_all_examples)
            cPickle.dump((_all_examples, cnt), myopen(training_examples_cache_filename(), "wb"), protocol=-1)
            assert len(_all_examples) == cnt
            logging.info("Wrote %d training examples to %s" % (cnt, training_examples_cache_filename()))
            logging.info(stats())
    assert _all_examples is not None
    return _all_examples

        

def get_all_training_examples_cached():
    for e in all_training_examples_cached():
        yield e
    
def get_training_minibatch_cached():
    import common.hyperparameters
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    MINIBATCH_SIZE = HYPERPARAMETERS["MINIBATCH SIZE"]

    minibatch = []
    for e in get_all_training_examples_cached():
        minibatch.append(e)
        if len(minibatch) >= MINIBATCH_SIZE:
            yield minibatch
            minibatch = []
    if len(minibatch) > 0:
        yield minibatch
        minibatch = []

if __name__ == "__main__":
    for minibatch in get_training_minibatch_cached():
#        print len(minibatch)
        for e in minibatch:
            print e

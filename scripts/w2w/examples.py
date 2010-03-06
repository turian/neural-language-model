"""
Streaming examples.
"""

from w2w.corpora import bicorpora_filenames, monocorpora_filenames, bicorpus_sentences_and_alignments
from common.file import myopen

from w2w.targetvocabulary import targetmap
from w2w.vocabulary import wordmap, language, wordform
import string
import logging

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
        return "%s" % `(self.l1, wordform(self.w1), [wordmap().str(w)[1] for w in self.l1seq], wordmap().str(self.w2))`

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

            # If we are filtering examples by lemma
            if not(HYPERPARAMETERS["W2W FOCUS LEMMAS"] is None or len (HYPERPARAMETERS["W2W FOCUS LEMMAS"]) == 0):
                assert language(w1) == "en"
                from lemmatizer import lemmatize
                if lemmatize(language(w1), wordform(w1)) not in HYPERPARAMETERS["W2W FOCUS LEMMAS"]:
                    logging.debug("Focus word %s (lemma %s) not in our list of focus lemmas" % (`wordmap().str(w1)`, lemmatize(language(w1), wordform(w1))))
                    continue
                    

            # Skip translations from unknown words
            if wordform(w1) == "*UNKNOWN*": continue

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

def get_training_minibatch():
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

if __name__ == "__main__":
    for minibatch in get_training_minibatch():
#        print len(minibatch)
        for e in minibatch:
            print e

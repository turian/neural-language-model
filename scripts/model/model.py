from parameters import Parameters

from hyperparameters import HYPERPARAMETERS
LBL = HYPERPARAMETERS["LOG BILINEAR MODEL"]

if LBL:
    import graphlbl as graph
else:
    import graphcw as graph

import sys, pickle
import math
import logging

from common.file import myopen
from common.movingaverage import MovingAverage

from vocabulary import *

class Model:
    """
    A Model can:

    @type parameters: L{Parameters}
    @todo: Document
    """

    def __init__(self):
        self.parameters = Parameters()
        if LBL:
            graph.output_weights = self.parameters.output_weights
            graph.output_biases = self.parameters.output_biases
            graph.score_biases = self.parameters.score_biases
        else:
            graph.hidden_weights = self.parameters.hidden_weights
            graph.hidden_biases = self.parameters.hidden_biases
            graph.output_weights = self.parameters.output_weights
            graph.output_biases = self.parameters.output_biases

#        (self.graph_train, self.graph_predict, self.graph_verbose_predict) = graph.functions(self.parameters)
        import sets
        self.train_loss = MovingAverage()
        self.train_err = MovingAverage()
        self.train_lossnonzero = MovingAverage()
        self.train_squashloss = MovingAverage()
        self.train_unpenalized_loss = MovingAverage()
        self.train_l1penalty = MovingAverage()
        self.train_unpenalized_lossnonzero = MovingAverage()
        self.train_cnt = 0

    def load(self, filename):
        sys.stderr.write("Loading model from: %s\n" % filename)
        f = myopen(filename, "rb")
        (self.parameters, self.train_loss, self.train_err, self.train_lossnonzero, self.train_squashloss, self.train_unpenalized_loss, self.train_l1penalty, self.train_unpenalized_lossnonzero, self.train_cnt) = pickle.load(f)

    def save(self, filename):
        sys.stderr.write("Saving model to: %s\n" % filename)
        f = myopen(filename, "wb")
        pickle.dump((self.parameters, self.train_loss, self.train_err, self.train_lossnonzero, self.train_squashloss, self.train_unpenalized_loss, self.train_l1penalty, self.train_unpenalized_lossnonzero, self.train_cnt), f)

    def embed(self, sequence):
        """
        Embed a sequence of vocabulary IDs
        """
        seq = [self.parameters.embeddings[s] for s in sequence]
        import numpy
        return [numpy.resize(s, (1, s.size)) for s in seq]
#        return [self.parameters.embeddings[s] for s in sequence]

    def corrupt_example(self, e):
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
                e[-1] = random.randint(0, self.parameters.vocab_size-1)
                pr = 1./self.parameters.vocab_size
            elif HYPERPARAMETERS["NGRAM_FOR_TRAINING_NOISE"] == 1:
                import noise
                from common.myrandom import weighted_sample
                e[-1], pr = weighted_sample(noise.indexed_weights())
#                from vocabulary import wordmap
#                print wordmap.str(e[-1]), pr
            else:
                assert 0
            cnt += 1
            # Backoff to 0gram smoothing if we fail 10 times to get noise.
            if cnt > 10: e[-1] = random.randint(0, self.parameters.vocab_size-1)
        weight = 1./pr
        return e, weight

    def train(self, correct_sequence):
        from hyperparameters import HYPERPARAMETERS
        learning_rate = HYPERPARAMETERS["LEARNING_RATE"]
        if LBL:
            noise_sequence, weight = self.corrupt_example(correct_sequence)
#            noise_repr = noise_sequence[-1]
#            correct_repr = correct_sequence[-1]
            noise_repr = noise_sequence[-1:]
            correct_repr = correct_sequence[-1:]
            assert noise_repr != correct_repr
            assert noise_sequence[:-1] == correct_sequence[:-1]
            sequence = correct_sequence[:-1]
#            r = graph.train(self.embed(sequence), self.embed([correct_repr])[0], self.embed([noise_repr])[0], self.parameters.score_biases[correct_repr], self.parameters.score_biases[noise_repr])
            r = graph.train(self.embed(sequence), self.embed(correct_repr)[0], self.embed(noise_repr)[0], self.parameters.score_biases[correct_repr], self.parameters.score_biases[noise_repr], learning_rate * weight)
            assert len(noise_repr) == 1
            assert len(correct_repr) == 1
            noise_repr = noise_repr[0]
            correct_repr = correct_repr[0]
            (loss, predictrepr, correct_score, noise_score, dsequence, dcorrect_repr, dnoise_repr, dcorrect_scorebias, dnoise_scorebias) = r
#            print
#            print "loss = ", loss
#            print "predictrepr = ", predictrepr
#            print "correct_repr = ", correct_repr, self.embed(correct_repr)[0]
#            print "noise_repr = ", noise_repr, self.embed(noise_repr)[0]
#            print "correct_score = ", correct_score
#            print "noise_score = ", noise_score
        else:
            noise_sequence, weight = self.corrupt_example(correct_sequence)
            r = graph.train(self.embed(correct_sequence), self.embed(noise_sequence), learning_rate * weight)
            (dcorrect_inputs, dnoise_inputs, loss, unpenalized_loss, l1penalty, correct_score, noise_score) = r
#            print unpenalized_loss, l1penalty, self.embed(correct_sequence), self.embed(noise_sequence)
#        print loss, correct_score, noise_score,
#        print loss, correct_score, noise_score
#        print ""
#        print "OLD: loss = %.3f, correct score = %.3f, noise score = %.3f" % (loss, correct_score, noise_score)
#        print "OLD: w1", self.parameters.hidden_weights
#        print "OLD: b1", self.parameters.hidden_biases
#        print "OLD: w2", self.parameters.output_weights
#        print "OLD: b2", self.parameters.output_biases
#        def st(x):
#            import numpy
#            s = ""
#            s += "avg=%.3f, std=%.3f" % (numpy.mean(x), numpy.std(x))
#            s += " "
#            s += "avgabs=%.3f, stdabs=%.3f" % (numpy.mean(numpy.abs(x)), numpy.std(numpy.abs(x)))
#            return s
#        print "OLD: w1", st(self.parameters.hidden_weights)
#        print "OLD: b1", st(self.parameters.hidden_biases)
#        print "OLD: w2", st(self.parameters.output_weights)
#        print "OLD: b2", st(self.parameters.output_biases)
#
#        if loss == 0:
#            print "OLD: dw1", st(dhidden_weights)
#            print "OLD: db1", st(dhidden_biases)
#            print "OLD: dw2", st(doutput_weights)
#            print "OLD: db2", st(doutput_biases)

        self.train_loss.add(loss)
        self.train_err.add(correct_score <= noise_score)
        self.train_lossnonzero.add(loss > 0)
        squashloss = 1./(1.+math.exp(-loss))
        self.train_squashloss.add(squashloss)
        if not LBL:
            self.train_unpenalized_loss.add(unpenalized_loss)
            self.train_l1penalty.add(l1penalty)
            self.train_unpenalized_lossnonzero.add(unpenalized_loss > 0)

        self.train_cnt += 1
        if self.train_cnt % 10000 == 0:
            logging.info(("After %d updates, pre-update train loss %s" % (self.train_cnt, self.train_loss.verbose_string())))
            logging.info(("After %d updates, pre-update train error %s" % (self.train_cnt, self.train_err.verbose_string())))
            logging.info(("After %d updates, pre-update train Pr(loss != 0) %s" % (self.train_cnt, self.train_lossnonzero.verbose_string())))
            logging.info(("After %d updates, pre-update train squash(loss) %s" % (self.train_cnt, self.train_squashloss.verbose_string())))
            if not LBL:
                logging.info(("After %d updates, pre-update train unpenalized loss %s" % (self.train_cnt, self.train_unpenalized_loss.verbose_string())))
                logging.info(("After %d updates, pre-update train l1penalty %s" % (self.train_cnt, self.train_l1penalty.verbose_string())))
                logging.info(("After %d updates, pre-update train Pr(unpenalized loss != 0) %s" % (self.train_cnt, self.train_unpenalized_lossnonzero.verbose_string())))

            if LBL:
                i = 1.
                while i < wordmap.len:
                    inti = int(i)
                    str = "word %s, rank %d, score %f" % (wordmap.str(inti), inti, self.parameters.score_biases[inti])
                    logging.info("After %d updates, score biases: %s" % (self.train_cnt, str))
                    i *= 3.2

#            print(("After %d updates, pre-update train loss %s" % (self.train_cnt, self.train_loss.verbose_string())))
#            print(("After %d updates, pre-update train error %s" % (self.train_cnt, self.train_err.verbose_string())))

        embedding_learning_rate = HYPERPARAMETERS["EMBEDDING_LEARNING_RATE"] * weight
        if loss == 0:
            if LBL:
                for di in dsequence + [dcorrect_repr, dnoise_repr]:
                    # This tends to trigger if training diverges (NaN)
                    assert (di == 0).all()
#                if not (di == 0).all():
#                    print "WARNING:", di
#                    print "WARNING in ", dsequence + [dcorrect_repr, dnoise_repr]
#                    print "loss = ", loss
#                    print "predictrepr = ", predictrepr
#                    print "correct_repr = ", correct_repr, self.embed(correct_repr)[0]
#                    print "noise_repr = ", noise_repr, self.embed(noise_repr)[0]
#                    print "correct_score = ", correct_score
#                    print "noise_score = ", noise_score
            else:
                for di in dcorrect_inputs + dnoise_inputs:
                    assert (di == 0).all()

        if loss != 0:
            import sets
            to_normalize = sets.Set()

            if LBL:
                val = sequence + [correct_repr, noise_repr]
                dval = dsequence + [dcorrect_repr, dnoise_repr]
#                print val
                for (i, di) in zip(val, dval):
#                for (i, di) in zip(tuple(sequence + [correct_repr, noise_repr]), tuple(dsequence + [dcorrect_repr, dnoise_repr])):
                    assert di.shape[0] == 1
                    di.resize(di.size)
#                    print i, di
                    self.parameters.embeddings[i] -= 1.0 * embedding_learning_rate * di
                    if HYPERPARAMETERS["NORMALIZE_EMBEDDINGS"]:
                        to_normalize.add(i)

                for (i, di) in zip([correct_repr, noise_repr], [dcorrect_scorebias, dnoise_scorebias]):
                    self.parameters.score_biases[i] -= 1.0 * embedding_learning_rate * di
#                    print "REMOVEME", i, self.parameters.score_biases[i]
            else:
                for (i, di) in zip(correct_sequence, dcorrect_inputs):
                    assert di.shape[0] == 1
                    di.resize(di.size)
#                    print i, di
                    self.parameters.embeddings[i] -= 1.0 * embedding_learning_rate * di
                    if HYPERPARAMETERS["NORMALIZE_EMBEDDINGS"]:
                        to_normalize.add(i)
                for (i, di) in zip(noise_sequence, dnoise_inputs):
                    assert di.shape[0] == 1
                    di.resize(di.size)
#                    print i, di
                    self.parameters.embeddings[i] -= 1.0 * embedding_learning_rate * di
                    if HYPERPARAMETERS["NORMALIZE_EMBEDDINGS"]:
                        to_normalize.add(i)
#                print to_normalize

            if len(to_normalize) > 0:
                to_normalize = [i for i in to_normalize]
#                print "NORMALIZING", to_normalize
                self.parameters.normalize(to_normalize)
    
#           r = graph.train(self.embed(correct_sequence), self.embed(noise_sequence), self.parameters)
#           (dcorrect_inputs, dnoise_inputs, loss, correct_score, noise_score, dhidden_weights, dhidden_biases, doutput_weights, doutput_biases) = r
#           print loss, correct_score, noise_score
#           print "NEW: loss = %.3f, correct score = %.3f, noise score = %.3f" % (loss, correct_score, noise_score)

#            r = graph.train(self.embed(sequence), self.embed(correct_repr)[0], self.embed(noise_repr)[0], self.parameters)
#            (loss, predictrepr, correct_score, noise_score, dsequence, dcorrect_repr, dnoise_repr, doutput_weights, doutput_biases) = r
#            print
#            print "loss = ", loss
#            print "predictrepr = ", predictrepr
#            print "correct_repr = ", correct_repr, self.embed(correct_repr)[0]
#            print "noise_repr = ", noise_repr, self.embed(noise_repr)[0]
#            print "correct_score = ", correct_score
#            print "noise_score = ", noise_score

    def predict(self, sequence):
        if LBL:
            targetrepr = sequence[-1:]
            sequence = sequence[:-1]
            (predictrepr, score) = graph.predict(self.embed(sequence), self.embed(targetrepr)[0], self.parameters.score_biases[targetrepr], self.parameters)
            return score
        else:
            (score) = graph.predict(self.embed(sequence), self.parameters)
            return score

    def verbose_predict(self, sequence):
        if LBL:
            assert 0
        else:
            (score, prehidden) = graph.verbose_predict(self.embed(sequence), self.parameters)
            return score, prehidden

    def validate(self, sequence):
        """
        Get the rank of this final word, as opposed to all other words in the vocabulary.
        """
        import random
        r = random.Random()
        r.seed(0)
        from hyperparameters import HYPERPARAMETERS

        import copy
        corrupt_sequence = copy.copy(sequence)
        rank = 1
        correct_score = self.predict(sequence)
#        print "CORRECT", correct_score, [wordmap.str(id) for id in sequence]
        for i in range(self.parameters.vocab_size):
            if r.random() > HYPERPARAMETERS["PERCENT OF NOISE EXAMPLES FOR VALIDATION LOGRANK"]: continue
            if i == sequence[-1]: continue
            corrupt_sequence[-1] = i
            corrupt_score = self.predict(corrupt_sequence)
            if correct_score <= corrupt_score:
#                print " CORRUPT", corrupt_score, [wordmap.str(id) for id in corrupt_sequence]
                rank += 1
        return rank

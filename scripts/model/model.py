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

    import hyperparameters
    import miscglobals
    import vocabulary
    def __init__(self, name="", window_size=HYPERPARAMETERS["WINDOW_SIZE"], vocab_size=vocabulary.wordmap().len, embedding_size=HYPERPARAMETERS["EMBEDDING_SIZE"], hidden_size=HYPERPARAMETERS["HIDDEN_SIZE"], seed=miscglobals.RANDOMSEED, initial_embeddings=None):
        self.name = name
        self.parameters = Parameters(window_size, vocab_size, embedding_size, hidden_size, seed, initial_embeddings)
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
        self.train_correct_score = MovingAverage()
        self.train_noise_score = MovingAverage()
        self.train_cnt = 0

    def __getstate__(self):
        return (self.parameters, self.train_loss, self.train_err, self.train_lossnonzero, self.train_squashloss, self.train_unpenalized_loss, self.train_l1penalty, self.train_unpenalized_lossnonzero, self.train_correct_score, self.train_noise_score, self.train_cnt)

    def __setstate__(self, state):
        (self.parameters, self.train_loss, self.train_err, self.train_lossnonzero, self.train_squashloss, self.train_unpenalized_loss, self.train_l1penalty, self.train_unpenalized_lossnonzero, self.train_correct_score, self.train_noise_score, self.train_cnt) = state
        if LBL:
            graph.output_weights = self.parameters.output_weights
            graph.output_biases = self.parameters.output_biases
            graph.score_biases = self.parameters.score_biases
        else:
            graph.hidden_weights = self.parameters.hidden_weights
            graph.hidden_biases = self.parameters.hidden_biases
            graph.output_weights = self.parameters.output_weights
            graph.output_biases = self.parameters.output_biases

#    def load(self, filename):
#        sys.stderr.write("Loading model from: %s\n" % filename)
#        f = myopen(filename, "rb")
#        (self.parameters, self.train_loss, self.train_err, self.train_lossnonzero, self.train_squashloss, self.train_unpenalized_loss, self.train_l1penalty, self.train_unpenalized_lossnonzero, self.train_correct_score, self.train_noise_score, self.train_cnt) = pickle.load(f)
#        if LBL:
#            graph.output_weights = self.parameters.output_weights
#            graph.output_biases = self.parameters.output_biases
#            graph.score_biases = self.parameters.score_biases
#        else:
#            graph.hidden_weights = self.parameters.hidden_weights
#            graph.hidden_biases = self.parameters.hidden_biases
#            graph.output_weights = self.parameters.output_weights
#            graph.output_biases = self.parameters.output_biases
#
#    def save(self, filename):
#        sys.stderr.write("Saving model to: %s\n" % filename)
#        f = myopen(filename, "wb")
#        pickle.dump((self.parameters, self.train_loss, self.train_err, self.train_lossnonzero, self.train_squashloss, self.train_unpenalized_loss, self.train_l1penalty, self.train_unpenalized_lossnonzero, self.train_correct_score, self.train_noise_score, self.train_cnt), f)

    def embed(self, sequence):
        """
        Embed a sequence of vocabulary IDs
        """
        seq = [self.parameters.embeddings[s] for s in sequence]
        import numpy
        return [numpy.resize(s, (1, s.size)) for s in seq]
#        return [self.parameters.embeddings[s] for s in sequence]

    def embeds(self, sequences):
        """
        Embed sequences of vocabulary IDs.
        If we are given a list of MINIBATCH lists of SEQLEN items, return a list of SEQLEN matrices of shape (MINIBATCH, EMBSIZE)
        """
        embs = []
        for sequence in sequences:
            embs.append(self.embed(sequence))

        for emb in embs: assert len(emb) == len(embs[0])

        new_embs = []
        for i in range(len(embs[0])):
            colembs = [embs[j][i] for j in range(len(embs))]
            import numpy
            new_embs.append(numpy.vstack(colembs))
            assert new_embs[-1].shape == (len(sequences), self.parameters.embedding_size)
        assert len(new_embs) == len(sequences[0])
        return new_embs

    def train(self, correct_sequences, noise_sequences, weights):
        from hyperparameters import HYPERPARAMETERS
        learning_rate = HYPERPARAMETERS["LEARNING_RATE"]

        # All weights must be the same, because of how we use a scalar learning rate
        assert HYPERPARAMETERS["UNIFORM EXAMPLE WEIGHTS"]
        if HYPERPARAMETERS["UNIFORM EXAMPLE WEIGHTS"]:
            for w in weights: assert w == weights[0]

        if LBL:
            # REWRITE FOR MINIBATCH
            assert 0

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
            r = graph.train(self.embeds(correct_sequences), self.embeds(noise_sequences), learning_rate * weights[0])
            (dcorrect_inputss, dnoise_inputss, losss, unpenalized_losss, l1penaltys, correct_scores, noise_scores) = r
#            print [d.shape for d in dcorrect_inputss]
#            print [d.shape for d in dnoise_inputss]
#            print "losss", losss.shape, losss
#            print "unpenalized_losss", unpenalized_losss.shape, unpenalized_losss
#            print "l1penaltys", l1penaltys.shape, l1penaltys
#            print "correct_scores", correct_scores.shape, correct_scores
#            print "noise_scores", noise_scores.shape, noise_scores

        import sets
        to_normalize = sets.Set()
        for ecnt in range(len(correct_sequences)):
            (loss, unpenalized_loss, correct_score, noise_score) = \
                (losss[ecnt], unpenalized_losss[ecnt], correct_scores[ecnt], noise_scores[ecnt])
            if l1penaltys.shape == ():
                assert l1penaltys == 0
                l1penalty = 0
            else:
                l1penalty = l1penaltys[ecnt]
            correct_sequence = correct_sequences[ecnt]
            noise_sequence = noise_sequences[ecnt]

            dcorrect_inputs = [d[ecnt] for d in dcorrect_inputss]
            dnoise_inputs = [d[ecnt] for d in dnoise_inputss]

#            print [d.shape for d in dcorrect_inputs]
#            print [d.shape for d in dnoise_inputs]
#            print "loss", loss.shape, loss
#            print "unpenalized_loss", unpenalized_loss.shape, unpenalized_loss
#            print "l1penalty", l1penalty.shape, l1penalty
#            print "correct_score", correct_score.shape, correct_score
#            print "noise_score", noise_score.shape, noise_score


            self.train_loss.add(loss)
            self.train_err.add(correct_score <= noise_score)
            self.train_lossnonzero.add(loss > 0)
            squashloss = 1./(1.+math.exp(-loss))
            self.train_squashloss.add(squashloss)
            if not LBL:
                self.train_unpenalized_loss.add(unpenalized_loss)
                self.train_l1penalty.add(l1penalty)
                self.train_unpenalized_lossnonzero.add(unpenalized_loss > 0)
            self.train_correct_score.add(correct_score)
            self.train_noise_score.add(noise_score)
    
            self.train_cnt += 1
            if self.train_cnt % 10000 == 0:
    #        if self.train_cnt % 1000 == 0:
    #            print self.train_cnt
#                graph.COMPILE_MODE.print_summary()
                logging.info(("After %d updates, pre-update train loss %s" % (self.train_cnt, self.train_loss.verbose_string())))
                logging.info(("After %d updates, pre-update train error %s" % (self.train_cnt, self.train_err.verbose_string())))
                logging.info(("After %d updates, pre-update train Pr(loss != 0) %s" % (self.train_cnt, self.train_lossnonzero.verbose_string())))
                logging.info(("After %d updates, pre-update train squash(loss) %s" % (self.train_cnt, self.train_squashloss.verbose_string())))
                if not LBL:
                    logging.info(("After %d updates, pre-update train unpenalized loss %s" % (self.train_cnt, self.train_unpenalized_loss.verbose_string())))
                    logging.info(("After %d updates, pre-update train l1penalty %s" % (self.train_cnt, self.train_l1penalty.verbose_string())))
                    logging.info(("After %d updates, pre-update train Pr(unpenalized loss != 0) %s" % (self.train_cnt, self.train_unpenalized_lossnonzero.verbose_string())))
                logging.info(("After %d updates, pre-update train correct score %s" % (self.train_cnt, self.train_correct_score.verbose_string())))
                logging.info(("After %d updates, pre-update train noise score %s" % (self.train_cnt, self.train_noise_score.verbose_string())))

                self.debug_prehidden_values(correct_sequences)
    
                if LBL:
                    i = 1.
                    while i < wordmap.len:
                        inti = int(i)
                        str = "word %s, rank %d, score %f" % (wordmap.str(inti), inti, self.parameters.score_biases[inti])
                        logging.info("After %d updates, score biases: %s" % (self.train_cnt, str))
                        i *= 3.2
    
    #            print(("After %d updates, pre-update train loss %s" % (self.train_cnt, self.train_loss.verbose_string())))
    #            print(("After %d updates, pre-update train error %s" % (self.train_cnt, self.train_err.verbose_string())))
    

            # All weights must be the same, because of how we use a scalar learning rate
            assert HYPERPARAMETERS["UNIFORM EXAMPLE WEIGHTS"]
            if HYPERPARAMETERS["UNIFORM EXAMPLE WEIGHTS"]:
                for w in weights: assert w == weights[0]
            embedding_learning_rate = HYPERPARAMETERS["EMBEDDING_LEARNING_RATE"] * weights[0]
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
#                        assert di.shape[0] == 1
#                        di.resize(di.size)
    #                    print i, di
                        assert di.shape == (self.parameters.embedding_size,)
                        self.parameters.embeddings[i] -= 1.0 * embedding_learning_rate * di
                        if HYPERPARAMETERS["NORMALIZE_EMBEDDINGS"]:
                            to_normalize.add(i)
                    for (i, di) in zip(noise_sequence, dnoise_inputs):
#                        assert di.shape[0] == 1
#                        di.resize(di.size)
    #                    print i, di
                        assert di.shape == (self.parameters.embedding_size,)
                        self.parameters.embeddings[i] -= 1.0 * embedding_learning_rate * di
                        if HYPERPARAMETERS["NORMALIZE_EMBEDDINGS"]:
                            to_normalize.add(i)
    #                print to_normalize
    
        if len(to_normalize) > 0:
            to_normalize = [i for i in to_normalize]
#            print "NORMALIZING", to_normalize
            self.parameters.normalize(to_normalize)



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
    
    def debug_prehidden_values(self, sequences):
        """
        Give debug output on pre-squash hidden values.
        """
        import numpy
        for (i, ve) in enumerate(sequences):
            (score, prehidden) = self.verbose_predict(ve)
            abs_prehidden = numpy.abs(prehidden)
            med = numpy.median(abs_prehidden)
            abs_prehidden = abs_prehidden.tolist()
            assert len(abs_prehidden) == 1
            abs_prehidden = abs_prehidden[0]
            abs_prehidden.sort()
            abs_prehidden.reverse()
            logging.info("model %s, %s %s %s %s %s" % (self.name, self.train_cnt, "abs(pre-squash hidden) median =", med, "max =", abs_prehidden[:3]))
            if i+1 >= 3: break

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


import graph
from parameters import Parameters

import hyperparameters

import sys, pickle
import math

from common.file import myopen

from vocabulary import *

class Model:
    """
    A Model can:

    @type parameters: L{Parameters}
    @todo: Document
    """

    def __init__(self):
        self.parameters = Parameters()

    def load(self, filename):
        sys.stderr.write("Loading model from: %s\n" % filename)
        f = myopen(filename, "rb")
        self.parameters = pickle.load(f)

    def save(self, filename):
        sys.stderr.write("Saving model to: %s\n" % filename)
        f = myopen(filename, "wb")
        pickle.dump(self.parameters, f)

    def embed(self, sequence):
        """
        Embed a sequence of vocabulary IDs
        """
        seq = [self.parameters.embeddings[s] for s in sequence]
        import numpy
        return [numpy.resize(s, (1, s.size)) for s in seq]
#        return [self.parameters.embeddings[s] for s in sequence]

    def corrupt_example(self, e):
        import random
        import copy
        e = copy.copy(e)
        last = e[-1]
        while e[-1] == last:
            e[-1] = random.randint(0, self.parameters.vocab_size-1)
        return e

    def train(self, correct_sequence):
        noise_sequence = self.corrupt_example(correct_sequence)
        r = graph.train(self.embed(correct_sequence), self.embed(noise_sequence), self.parameters)
        (dcorrect_inputs, dnoise_inputs, loss, correct_score, noise_score, dhidden_weights, dhidden_biases, doutput_weights, doutput_biases) = r
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

        self.parameters.hidden_weights   -= 1.0 * hyperparameters.LEARNING_RATE * dhidden_weights
        self.parameters.hidden_biases    -= 1.0 * hyperparameters.LEARNING_RATE * dhidden_biases
        self.parameters.output_weights   -= 1.0 * hyperparameters.LEARNING_RATE * doutput_weights
        self.parameters.output_biases    -= 1.0 * hyperparameters.LEARNING_RATE * doutput_biases

        for (i, di) in zip(correct_sequence, correct_score):
            self.parameters.embeddings[i] -= 1.0 * hyperparameters.LEARNING_RATE * di
        for (i, di) in zip(noise_sequence, noise_score):
            self.parameters.embeddings[i] -= 1.0 * hyperparameters.LEARNING_RATE * di

#        r = graph.train(self.embed(correct_sequence), self.embed(noise_sequence), self.parameters)
#        (dcorrect_inputs, dnoise_inputs, loss, correct_score, noise_score, dhidden_weights, dhidden_biases, doutput_weights, doutput_biases) = r
#        print loss, correct_score, noise_score
#        print "NEW: loss = %.3f, correct score = %.3f, noise score = %.3f" % (loss, correct_score, noise_score)

    def predict(self, sequence):
        (score) = graph.predict(self.embed(sequence), self.parameters)
        return score

    def verbose_predict(self, sequence):
        (score, prehidden) = graph.verbose_predict(self.embed(sequence), self.parameters)
        return score, prehidden

    def validate(self, sequence):
        """
        Get the rank of this final word, as opposed to all other words in the vocabulary.
        """
        import copy
        corrupt_sequence = copy.copy(sequence)
        rank = 1
        correct_score = self.predict(sequence)
#        print "CORRECT", correct_score, [wordmap.str(id) for id in sequence]
        for i in range(self.parameters.vocab_size):
            if i == sequence[-1]: continue
            corrupt_sequence[-1] = i
            corrupt_score = self.predict(corrupt_sequence)
            if correct_score <= corrupt_score:
#                print " CORRUPT", corrupt_score, [wordmap.str(id) for id in corrupt_sequence]
                rank += 1
        return rank

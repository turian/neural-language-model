
import graph
from parameters import Parameters

import hyperparameters

import sys, pickle
import math

from common.file import myopen

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

    def train(self, correct_sequence, noise_sequence):
        r = graph.train(self.embed(correct_sequence), self.embed(noise_sequence), self.parameters)
        (loss, correct_score, noise_score, dhidden_weights, dhidden_biases, doutput_weights, doutput_biases) = r
        print loss, correct_score, noise_score

        self.parameters.hidden_weights   -= 1.0 * hyperparameters.LEARNING_RATE * dhidden_weights
        self.parameters.hidden_biases    -= 1.0 * hyperparameters.LEARNING_RATE * dhidden_biases
        self.parameters.output_weights   -= 1.0 * hyperparameters.LEARNING_RATE * doutput_weights
        self.parameters.output_biases    -= 1.0 * hyperparameters.LEARNING_RATE * doutput_biases

        r = graph.train(self.embed(correct_sequence), self.embed(noise_sequence), self.parameters)
        (loss, correct_score, noise_score, dhidden_weights, dhidden_biases, doutput_weights, doutput_biases) = r
        print loss, correct_score, noise_score

    def predict(self, sequence):
        (score) = graph.predict(self.embed(sequence), self.parameters)
        return score


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
        return [self.parameters.embeddings[s] for s in sequence]

    def train(self, sequence, target_output):
        assert not hyperparameters.TUNE_EMBEDDINGS

#        if self.parameters.hidden_layers == 1:
        (loss, argmax_class, max_pr, dhidden_weights, dhidden_biases, doutput_weights, doutput_biases) = graph.train(sequence, target_output, self.parameters)
#        elif self.parameters.hidden_layers == 2:
#            (loss, argmax_class, max_pr, dhidden_weights, dhidden_biases, dhidden2_weights, dhidden2_biases, doutput_weights, doutput_biases) = graph.train(sequence, target_output, self.parameters)
#        else:
#            assert 0
        assert len(argmax_class) == 1 and len(max_pr) == 1
#        (argmax_class, max_pr) = (argmax_class[0], max_pr[0])
#        print "Old loss: %.3f (Pr(%s) = %.3f" % (loss, labelmap.str(target_output), math.exp(-loss)),
#        if argmax_class == target_output: print "[MAX])"
#        else: print "vs. Pr(%s) = %.3f)" % (labelmap.str(argmax_class), max_pr)
        if hyperparameters.DIVIDE_LEARNING_RATE_BY_NUMBER_OF_INPUTS:
            self.parameters.hidden_weights   -= 1.0 * hyperparameters.LEARNING_RATE / self.parameters.total_input_dimension * dhidden_weights
            self.parameters.hidden_biases    -= 1.0 * hyperparameters.LEARNING_RATE / self.parameters.total_input_dimension * dhidden_biases
#            if self.parameters.hidden_layers == 2:
#                self.parameters.hidden2_weights   -= 1.0 * hyperparameters.LEARNING_RATE / self.parameters.embedding_size * dhidden2_weights
#                self.parameters.hidden2_biases    -= 1.0 * hyperparameters.LEARNING_RATE / self.parameters.embedding_size * dhidden2_biases
            self.parameters.output_weights   -= 1.0 * hyperparameters.LEARNING_RATE / self.parameters.embedding_size * doutput_weights
            self.parameters.output_biases    -= 1.0 * hyperparameters.LEARNING_RATE / self.parameters.embedding_size * doutput_biases
#            self.parameters.hidden_weights   -= 1.0 * hyperparameters.LEARNING_RATE / self.parameters.embedding_size * dhidden_weights
#            self.parameters.hidden_biases    -= 1.0 * hyperparameters.LEARNING_RATE / self.parameters.embedding_size * dhidden_biases
#            self.parameters.output_weights   -= 1.0 * hyperparameters.LEARNING_RATE / self.parameters.output_vocabsize * doutput_weights
#            self.parameters.output_biases    -= 1.0 * hyperparameters.LEARNING_RATE / self.parameters.output_vocabsize * doutput_biases
        else:
            self.parameters.hidden_weights   -= 1.0 * hyperparameters.LEARNING_RATE * dhidden_weights
            self.parameters.hidden_biases    -= 1.0 * hyperparameters.LEARNING_RATE * dhidden_biases
#            if self.parameters.hidden_layers == 2:
#                self.parameters.hidden2_weights   -= 1.0 * hyperparameters.LEARNING_RATE * dhidden2_weights
#                self.parameters.hidden2_biases    -= 1.0 * hyperparameters.LEARNING_RATE * dhidden2_biases
            self.parameters.output_weights   -= 1.0 * hyperparameters.LEARNING_RATE * doutput_weights
            self.parameters.output_biases    -= 1.0 * hyperparameters.LEARNING_RATE * doutput_biases
#        (loss, argmax_class, max_pr) = graph.predict(sequence, target_output, self.parameters)
#        assert len(argmax_class) == 1 and len(max_pr) == 1
#        (argmax_class, max_pr) = (argmax_class[0], max_pr[0])
#        print "New loss: %.3f (Pr(%s) = %.3f" % (loss, labelmap.str(target_output), math.exp(-loss)),
#        if argmax_class == target_output: print "[MAX])"
#        else: print "vs. Pr(%s) = %.3f)" % (labelmap.str(argmax_class), max_pr)

    def predict(self, sequence):
        (score) = graph.predict(self.embed(sequence), self.parameters)
        return score

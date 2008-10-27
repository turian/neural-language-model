
import graph
from parameters import Parameters

import hyperparameters

import sys, pickle
import math

from common.file import myopen
from vocabularies import *

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

    def learn(self, sequence, target_output):
        assert not hyperparameters.TUNE_EMBEDDINGS

        if self.parameters.hidden_layers == 1:
            (loss, argmax_class, max_pr, dconvolution_weights, dconvolution_biases, dunembedding_weights, dunembedding_biases) = graph.learn(sequence, target_output, self.parameters)
        elif self.parameters.hidden_layers == 2:
            (loss, argmax_class, max_pr, dconvolution_weights, dconvolution_biases, dhidden2_weights, dhidden2_biases, dunembedding_weights, dunembedding_biases) = graph.learn(sequence, target_output, self.parameters)
        else:
            assert 0
        assert len(argmax_class) == 1 and len(max_pr) == 1
#        (argmax_class, max_pr) = (argmax_class[0], max_pr[0])
#        print "Old loss: %.3f (Pr(%s) = %.3f" % (loss, labelmap.str(target_output), math.exp(-loss)),
#        if argmax_class == target_output: print "[MAX])"
#        else: print "vs. Pr(%s) = %.3f)" % (labelmap.str(argmax_class), max_pr)
        if hyperparameters.DIVIDE_LEARNING_RATE_BY_NUMBER_OF_INPUTS:
            self.parameters.convolution_weights   -= 1.0 * hyperparameters.LEARNING_RATE / self.parameters.total_input_dimension * dconvolution_weights
            self.parameters.convolution_biases    -= 1.0 * hyperparameters.LEARNING_RATE / self.parameters.total_input_dimension * dconvolution_biases
            if self.parameters.hidden_layers == 2:
                self.parameters.hidden2_weights   -= 1.0 * hyperparameters.LEARNING_RATE / self.parameters.embedding_size * dhidden2_weights
                self.parameters.hidden2_biases    -= 1.0 * hyperparameters.LEARNING_RATE / self.parameters.embedding_size * dhidden2_biases
            self.parameters.unembedding_weights   -= 1.0 * hyperparameters.LEARNING_RATE / self.parameters.embedding_size * dunembedding_weights
            self.parameters.unembedding_biases    -= 1.0 * hyperparameters.LEARNING_RATE / self.parameters.embedding_size * dunembedding_biases
#            self.parameters.convolution_weights   -= 1.0 * hyperparameters.LEARNING_RATE / self.parameters.embedding_size * dconvolution_weights
#            self.parameters.convolution_biases    -= 1.0 * hyperparameters.LEARNING_RATE / self.parameters.embedding_size * dconvolution_biases
#            self.parameters.unembedding_weights   -= 1.0 * hyperparameters.LEARNING_RATE / self.parameters.output_vocabsize * dunembedding_weights
#            self.parameters.unembedding_biases    -= 1.0 * hyperparameters.LEARNING_RATE / self.parameters.output_vocabsize * dunembedding_biases
        else:
            self.parameters.convolution_weights   -= 1.0 * hyperparameters.LEARNING_RATE * dconvolution_weights
            self.parameters.convolution_biases    -= 1.0 * hyperparameters.LEARNING_RATE * dconvolution_biases
            if self.parameters.hidden_layers == 2:
                self.parameters.hidden2_weights   -= 1.0 * hyperparameters.LEARNING_RATE * dhidden2_weights
                self.parameters.hidden2_biases    -= 1.0 * hyperparameters.LEARNING_RATE * dhidden2_biases
            self.parameters.unembedding_weights   -= 1.0 * hyperparameters.LEARNING_RATE * dunembedding_weights
            self.parameters.unembedding_biases    -= 1.0 * hyperparameters.LEARNING_RATE * dunembedding_biases
#        (loss, argmax_class, max_pr) = graph.predict(sequence, target_output, self.parameters)
#        assert len(argmax_class) == 1 and len(max_pr) == 1
#        (argmax_class, max_pr) = (argmax_class[0], max_pr[0])
#        print "New loss: %.3f (Pr(%s) = %.3f" % (loss, labelmap.str(target_output), math.exp(-loss)),
#        if argmax_class == target_output: print "[MAX])"
#        else: print "vs. Pr(%s) = %.3f)" % (labelmap.str(argmax_class), max_pr)

    def predict(self, sequence, target_output):
        (loss, argmax_class, max_pr) = graph.predict(sequence, target_output, self.parameters)
        assert len(argmax_class) == 1 and len(max_pr) == 1
        (argmax_class, max_pr) = (argmax_class[0], max_pr[0])
        return (loss, argmax_class, max_pr)

    def predict_all(self, sequence):
        (all_pr,) = graph.predict_all(sequence, self.parameters)
        assert all_pr.shape == (1, self.parameters.output_vocabsize)
        return all_pr[0]

    def validate(self, sequence, target_output):
#        o = graph.validate(sequence, target_output, self.parameters)
#        (loss, argmax_class, max_pr) = o[0:3]
#        neuron_outputs = o[3:]
        if self.parameters.hidden_layers == 2:
            (loss, argmax_class, max_pr, hidden1_layer_output, hidden2_layer_output) = graph.validate(sequence, target_output, self.parameters)
            assert hidden2_layer_output.shape == (1, self.parameters.embedding_size)
        else:
            assert self.parameters.hidden_layers == 1
            (loss, argmax_class, max_pr, hidden1_layer_output) = graph.validate(sequence, target_output, self.parameters)
        assert hidden1_layer_output.shape == (1, self.parameters.embedding_size)

        assert len(argmax_class) == 1 and len(max_pr) == 1
        (argmax_class, max_pr) = (argmax_class[0], max_pr[0])
        if self.parameters.hidden_layers == 2:
            return (loss, argmax_class, max_pr, hidden1_layer_output, hidden2_layer_output)
        else:
            return (loss, argmax_class, max_pr, hidden1_layer_output)

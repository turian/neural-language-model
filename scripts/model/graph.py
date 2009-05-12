import theano

from theano import tensor as t
from theano import scalar as s

from theano.tensor.basic import horizontal_stack
from theano.tensor import dot

from theano import gradient

import theano.compile
#from miscglobals import LINKER, OPTIMIZER
#mode = theano.compile.Mode(LINKER, OPTIMIZER)
#COMPILE_MODE = theano.compile.Mode('c|py', 'fast_run')
COMPILE_MODE = theano.compile.Mode('py', 'fast_compile')

import numpy

import hyperparameters

# TODO: Pure stochastic one-example-per training?
# TODO: Reuse subfunctions, e.g. autoassociators, instead of rewriting them
# TODO: DESCRIBEME: Better var names?

hidden_weights = t.dmatrix()
hidden_biases = t.dmatrix()

#if hyperparameters.USE_SECOND_HIDDEN_LAYER == True:
#    hidden2_weights = t.dmatrix()
#    hidden2_biases = t.dmatrix()

output_weights = t.dmatrix()
output_biases = t.dmatrix()

# TODO: Include gradient steps in actual function, don't do them manually

def activation_function(r):
    if hyperparameters.ACTIVATION_FUNCTION == "sigmoid":
        return sigmoid(r)
    elif hyperparameters.ACTIVATION_FUNCTION == "tanh":
        return t.tanh(r)
    elif hyperparameters.ACTIVATION_FUNCTION == "softsign":
        from theano.sandbox.softsign import softsign
        return softsign(r)
    else:
        assert 0

def stack(x):
    """
    Horizontally stack a list of representations, and then compress them to
    one representation.
    """
    assert len(x) >= 2
    return horizontal_stack(*x)

def score(inputs):
    x = stack(inputs)
    hidden = activation_function(dot(x, hidden_weights) + hidden_biases)
    score = dot(x, output_weights) + output_biases
    return score

cached_functions = {}
def functions(sequence_length):
    """
    Return two functions
     * The first function does prediction.
     * The second function does learning.
    """
    p = (sequence_length)
    if p not in cached_functions:
        print "Need to construct graph for sequence_length=%d..." % (sequence_length)
        # Create the sequence_length inputs.
        # Each is a t.dmatrix(), initial word embeddings (provided by
        # Jason + Ronan) to be transformed into an initial representation.
        # We could use a vector, but instead we use a matrix with one row.
        correct_inputs = [t.dmatrix() for i in range(sequence_length)]
        noise_inputs = [t.dmatrix() for i in range(sequence_length)]

        correct_score = score(correct_inputs)
        noise_score = score(noise_inputs)
        loss = max(0, 1 - correct_score + noise_score)

        (dhidden_weights, dhidden_biases, doutput_weights, doutput_biases) = t.grad(loss, [hidden_weights, hidden_biases, output_weights, output_biases])
        predict_inputs = correct_inputs + [hidden_weights, hidden_biases, output_weights, output_biases]
        train_inputs = correct_inputs + noise_inputs + [hidden_weights, hidden_biases, output_weights, output_biases]
        predict_outputs = [correct_score]
        train_outputs = [loss, correct_score, noise_score, dhidden_weights, dhidden_biases, doutput_weights, doutput_biases]

        import theano.gof.graph

        nnodes = len(theano.gof.graph.ops(predict_inputs, predict_outputs))
        print "About to compile predict function over %d ops [nodes]..." % nnodes
        predict_function = theano.function(predict_inputs, predict_outputs, mode=COMPILE_MODE)
        print "...done constructing graph for sequence_length=%d" % (sequence_length)

        nnodes = len(theano.gof.graph.ops(train_inputs, train_outputs))
        print "About to compile train function over %d ops [nodes]..." % nnodes
        train_function = theano.function(train_inputs, train_outputs, mode=COMPILE_MODE)
        print "...done constructing graph for sequence_length=%d" % (sequence_length)

        cached_functions[p] = (predict_function, train_function)
    return cached_functions[p]

#def apply_function(fn, sequence, target_output, parameters):
#    assert len(sequence) == parameters.hidden_width
#    inputs = [numpy.asarray([token]) for token in sequence]
#    if target_output != None:
##        if hyperparameters.USE_SECOND_HIDDEN_LAYER:
##            return fn(*(inputs + [numpy.asarray([target_output]), parameters.hidden_weights, parameters.hidden_biases, parameters.hidden2_weights, parameters.hidden2_biases, parameters.output_weights, parameters.output_biases]))
##        else:
#        return fn(*(inputs + [numpy.asarray([target_output]), parameters.hidden_weights, parameters.hidden_biases, parameters.output_weights, parameters.output_biases]))
#    else:
##        if hyperparameters.USE_SECOND_HIDDEN_LAYER:
##            return fn(*(inputs + [parameters.hidden_weights, parameters.hidden_biases, parameters.hidden2_weights, parameters.hidden2_biases, parameters.output_weights, parameters.output_biases]))
##        else:
#        return fn(*(inputs + [parameters.hidden_weights, parameters.hidden_biases, parameters.output_weights, parameters.output_biases]))
#
def predict(correct_sequence, parameters):
    fn = functions(sequence_length=len(correct_sequence))[0]
    for i in (correct_sequence + [parameters.hidden_weights, parameters.hidden_biases, parameters.output_weights, parameters.output_biases]):
        print i.shape
    return fn(*(correct_sequence + [parameters.hidden_weights, parameters.hidden_biases, parameters.output_weights, parameters.output_biases]))
def train(correct_sequence, noise_sequence, parameters):
    assert len(correct_sequence) == len(noise_sequence)
    fn = functions(sequence_length=len(correct_sequence))[1]
    return fn(*(correct_sequence + noise_sequence + [parameters.hidden_weights, parameters.hidden_biases, parameters.output_weights, parameters.output_biases]))

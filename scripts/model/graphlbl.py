"""
Theano graph of Mnih log bi-linear model.
"""

import theano
import theano.sandbox.cuda
theano.sandbox.cuda.use()

from theano import tensor as t
from theano import scalar as s

from theano.tensor.basic import horizontal_stack
from theano.tensor import dot

from theano import gradient

import theano.compile
#from miscglobals import LINKER, OPTIMIZER
#mode = theano.compile.Mode(LINKER, OPTIMIZER)
COMPILE_MODE = theano.compile.Mode('c|py', 'fast_run')
#COMPILE_MODE = theano.compile.Mode('py', 'fast_compile')

import numpy

from common.chopargs import chopargs

#output_weights = t.xmatrix()
#output_biases = t.xmatrix()

# TODO: Include gradient steps in actual function, don't do them manually

def activation_function(r):
    from hyperparameters import HYPERPARAMETERS
    if HYPERPARAMETERS["ACTIVATION_FUNCTION"] == "sigmoid":
        return sigmoid(r)
    elif HYPERPARAMETERS["ACTIVATION_FUNCTION"] == "tanh":
        return t.tanh(r)
    elif HYPERPARAMETERS["ACTIVATION_FUNCTION"] == "softsign":
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

def score(targetrepr, predictrepr):
    # TODO: Is this the right scoring function?
    score = dot(targetrepr, predictrepr.T)
    return score

cached_functions = {}
def functions(sequence_length):
    """
    Return two functions
     * The first function does prediction.
     * The second function does learning.
    """
    global cached_functions
    p = (sequence_length)
    if len(cached_functions.keys()) > 1:
        # This is problematic because we use global variables for the model parameters.
        # Hence, we might be unsafe, if we are using the wrong model parameters globally.
        assert 0
    if p not in cached_functions:
        print "Need to construct graph for sequence_length=%d..." % (sequence_length)
        # Create the sequence_length inputs.
        # Each is a t.xmatrix(), initial word embeddings (provided by
        # Jason + Ronan) to be transformed into an initial representation.
        # We could use a vector, but instead we use a matrix with one row.
        sequence = [t.xmatrix() for i in range(sequence_length)]
        correct_repr = t.xmatrix()
        noise_repr = t.xmatrix()
#        correct_scorebias = t.xscalar()
#        noise_scorebias = t.xscalar()
        correct_scorebias = t.xvector()
        noise_scorebias = t.xvector()

        stackedsequence = stack(sequence)
        predictrepr = dot(stackedsequence, output_weights) + output_biases

        correct_score = score(correct_repr, predictrepr) + correct_scorebias
        noise_score = score(noise_repr, predictrepr) + noise_scorebias
        loss = t.clip(1 - correct_score + noise_score, 0, 1e999)

        (doutput_weights, doutput_biases) = t.grad(loss, [output_weights, output_biases])
        dsequence = t.grad(loss, sequence)
        (dcorrect_repr, dnoise_repr) = t.grad(loss, [correct_repr, noise_repr])
        (dcorrect_scorebias, dnoise_scorebias) = t.grad(loss, [correct_scorebias, noise_scorebias])
        #print "REMOVEME", len(dcorrect_inputs)
        predict_inputs = sequence + [correct_repr, correct_scorebias, output_weights, output_biases]
        train_inputs = sequence + [correct_repr, noise_repr, correct_scorebias, noise_scorebias, output_weights, output_biases]
        predict_outputs = [predictrepr, correct_score]
        train_outputs = [loss, predictrepr, correct_score, noise_score] + dsequence + [dcorrect_repr, dnoise_repr, doutput_weights, doutput_biases, dcorrect_scorebias, dnoise_scorebias]
#        train_outputs = [loss, correct_repr, correct_score, noise_repr, noise_score]

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
##        if HYPERPARAMETERS["USE_SECOND_HIDDEN_LAYER"]:
##            return fn(*(inputs + [numpy.asarray([target_output]), parameters.hidden_weights, parameters.hidden_biases, parameters.hidden2_weights, parameters.hidden2_biases, parameters.output_weights, parameters.output_biases]))
##        else:
#        return fn(*(inputs + [numpy.asarray([target_output]), parameters.hidden_weights, parameters.hidden_biases, parameters.output_weights, parameters.output_biases]))
#    else:
##        if HYPERPARAMETERS["USE_SECOND_HIDDEN_LAYER"]:
##            return fn(*(inputs + [parameters.hidden_weights, parameters.hidden_biases, parameters.hidden2_weights, parameters.hidden2_biases, parameters.output_weights, parameters.output_biases]))
##        else:
#        return fn(*(inputs + [parameters.hidden_weights, parameters.hidden_biases, parameters.output_weights, parameters.output_biases]))
#

def predict(sequence, targetrepr, target_scorebias):
    fn = functions(sequence_length=len(sequence))[0]
    (predictrepr, score) = fn(*(sequence + [targetrepr, target_scorebias]))
    return predictrepr, score

def train(sequence, correct_repr, noise_repr, correct_scorebias, noise_scorebias, learning_rate):
    fn = functions(sequence_length=len(sequence))[1]
#    print "REMOVEME", correct_scorebias, noise_scorebias
#    print "REMOVEME", correct_scorebias[0], noise_scorebias[0]
    r = fn(*(sequence + [correct_repr, noise_repr, correct_scorebias, noise_scorebias]))

    (loss, predictrepr, correct_score, noise_score, dsequence, dcorrect_repr, dnoise_repr, doutput_weights, doutput_biases, dcorrect_scorebias, dnoise_scorebias) = chopargs(r, (0,0,0,0,len(sequence),0,0,0,0,0,0))
    if loss == 0:
        for di in [doutput_weights, doutput_biases]:
            # This tends to trigger if training diverges (NaN)
            assert (di == 0).all()

    parameters.output_weights   -= 1.0 * learning_rate * doutput_weights
    parameters.output_biases    -= 1.0 * learning_rate * doutput_biases

    # You also need to update score_biases here
    assert 0

    dsequence = list(dsequence)
    return (loss, predictrepr, correct_score, noise_score, dsequence, dcorrect_repr, dnoise_repr, dcorrect_scorebias, dnoise_scorebias)

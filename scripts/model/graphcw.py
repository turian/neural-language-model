"""
Theano graph of Collobert & Weston language model.
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
import theano.compile.debugmode
#COMPILE_MODE = theano.compile.debugmode.DebugMode(optimizer='fast_run', check_isfinite=False)
COMPILE_MODE = theano.compile.Mode('c|py', 'fast_run')
#COMPILE_MODE = theano.compile.Mode('py', 'fast_compile')

import numpy

hidden_weights = t.xmatrix()
hidden_biases = t.xmatrix()

#if HYPERPARAMETERS["USE_SECOND_HIDDEN_LAYER"] == True:
#    hidden2_weights = t.xmatrix()
#    hidden2_biases = t.xmatrix()

output_weights = t.xmatrix()
output_biases = t.xmatrix()

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

def score(x):
    prehidden = dot(x, hidden_weights) + hidden_biases
    hidden = activation_function(prehidden)
    score = dot(hidden, output_weights) + output_biases
    return score, prehidden

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
        # Each is a t.xmatrix(), initial word embeddings (provided by
        # Jason + Ronan) to be transformed into an initial representation.
        # We could use a vector, but instead we use a matrix with one row.
        correct_inputs = [t.xmatrix() for i in range(sequence_length)]
        noise_inputs = [t.xmatrix() for i in range(sequence_length)]

        stacked_correct_inputs = stack(correct_inputs)
        stacked_noise_inputs = stack(noise_inputs)

        correct_score, correct_prehidden = score(stacked_correct_inputs)
        noise_score, noise_prehidden = score(stacked_noise_inputs)
        unpenalized_loss = t.clip(1 - correct_score + noise_score, 0, 1e999)

        from hyperparameters import HYPERPARAMETERS
        if HYPERPARAMETERS["CW_EMBEDDING_L1_PENALTY"] != 0:
            l1penalty = t.sum(t.abs_(stacked_correct_inputs) + t.abs_(stacked_noise_inputs)) * HYPERPARAMETERS["CW_EMBEDDING_L1_PENALTY"]
        else:
            l1penalty = t.as_tensor_variable(0)
        loss = unpenalized_loss + l1penalty

        (dhidden_weights, dhidden_biases, doutput_weights, doutput_biases) = t.grad(loss, [hidden_weights, hidden_biases, output_weights, output_biases])
        dcorrect_inputs = t.grad(loss, correct_inputs)
        dnoise_inputs = t.grad(loss, noise_inputs)
        #print "REMOVEME", len(dcorrect_inputs)
        predict_inputs = correct_inputs + [hidden_weights, hidden_biases, output_weights, output_biases]
        train_inputs = correct_inputs + noise_inputs + [hidden_weights, hidden_biases, output_weights, output_biases]
        verbose_predict_inputs = predict_inputs
        predict_outputs = [correct_score]
        train_outputs = dcorrect_inputs + dnoise_inputs + [loss, unpenalized_loss, l1penalty, correct_score, noise_score, dhidden_weights, dhidden_biases, doutput_weights, doutput_biases]
        verbose_predict_outputs = [correct_score, correct_prehidden]

        import theano.gof.graph

        nnodes = len(theano.gof.graph.ops(predict_inputs, predict_outputs))
        print "About to compile predict function over %d ops [nodes]..." % nnodes
        predict_function = theano.function(predict_inputs, predict_outputs, mode=COMPILE_MODE)
        print "...done constructing graph for sequence_length=%d" % (sequence_length)

        nnodes = len(theano.gof.graph.ops(verbose_predict_inputs, verbose_predict_outputs))
        print "About to compile predict function over %d ops [nodes]..." % nnodes
        verbose_predict_function = theano.function(verbose_predict_inputs, verbose_predict_outputs, mode=COMPILE_MODE)
        print "...done constructing graph for sequence_length=%d" % (sequence_length)

        nnodes = len(theano.gof.graph.ops(train_inputs, train_outputs))
        print "About to compile train function over %d ops [nodes]..." % nnodes
        train_function = theano.function(train_inputs, train_outputs, mode=COMPILE_MODE)
        print "...done constructing graph for sequence_length=%d" % (sequence_length)

        cached_functions[p] = (predict_function, train_function, verbose_predict_function)
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
def predict(correct_sequence, parameters):
    fn = functions(sequence_length=len(correct_sequence))[0]
    r = fn(*(correct_sequence + [parameters.hidden_weights, parameters.hidden_biases, parameters.output_weights, parameters.output_biases]))
    assert len(r) == 1
    r = r[0]
    assert r.shape == (1, 1)
    return r[0,0]
def verbose_predict(correct_sequence, parameters):
    fn = functions(sequence_length=len(correct_sequence))[2]
    r = fn(*(correct_sequence + [parameters.hidden_weights, parameters.hidden_biases, parameters.output_weights, parameters.output_biases]))
    assert len(r) == 2
    (score, prehidden) = r
    assert score.shape == (1, 1)
    return score[0,0], prehidden
def train(correct_sequence, noise_sequence, parameters):
    assert len(correct_sequence) == len(noise_sequence)
    fn = functions(sequence_length=len(correct_sequence))[1]
    r = fn(*(correct_sequence + noise_sequence + [parameters.hidden_weights, parameters.hidden_biases, parameters.output_weights, parameters.output_biases]))
    dcorrect_inputs = r[:len(correct_sequence)]
    r = r[len(correct_sequence):]
    dnoise_inputs = r[:len(noise_sequence)]
    r = r[len(correct_sequence):]
#    print "REMOVEME", len(dcorrect_inputs), len(dnoise_inputs)
    (loss, unpenalized_loss, l1penalty, correct_score, noise_score, dhidden_weights, dhidden_biases, doutput_weights, doutput_biases) = r
    return (dcorrect_inputs, dnoise_inputs, loss, unpenalized_loss, l1penalty, correct_score, noise_score, dhidden_weights, dhidden_biases, doutput_weights, doutput_biases)

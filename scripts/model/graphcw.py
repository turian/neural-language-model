"""
Theano graph of Collobert & Weston language model.
"""

import theano
#import theano.sandbox.cuda
#theano.sandbox.cuda.use()

from theano.compile import pfunc, shared
from theano import config
floatX = config.floatX


from theano import tensor as t
from theano import scalar as s

from theano.tensor.basic import horizontal_stack
from theano.tensor import dot

from theano import gradient

import theano.compile
#from miscglobals import LINKER, OPTIMIZER
#mode = theano.compile.Mode(LINKER, OPTIMIZER)
#import theano.compile.debugmode
#COMPILE_MODE = theano.compile.debugmode.DebugMode(optimizer='fast_run', check_isfinite=False)
#import theano.compile.profilemode
#COMPILE_MODE = theano.compile.profilemode.ProfileMode()
COMPILE_MODE = theano.compile.Mode('c|py', 'fast_run')
#COMPILE_MODE = theano.compile.Mode('py', 'fast_compile')

import numpy

#hidden_weights = t.matrix()
#hidden_biases = t.matrix()

#if HYPERPARAMETERS["USE_SECOND_HIDDEN_LAYER"] == True:
#    hidden2_weights = t.matrix()
#    hidden2_biases = t.matrix()

#output_weights = t.matrix()
#output_biases = t.matrix()

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
    global cached_functions
    cachekey = (sequence_length)
    if len(cached_functions.keys()) > 1:
        # This is problematic because we use global variables for the model parameters.
        # Hence, we might be unsafe, if we are using the wrong model parameters globally.
        assert 0
    if cachekey not in cached_functions:
        print "Need to construct graph for sequence_length=%d..." % (sequence_length)
        # Create the sequence_length inputs.
        # Each is a t.matrix(), initial word embeddings (provided by
        # Jason + Ronan) to be transformed into an initial representation.
        # We could use a vector, but instead we use a matrix with one row.
        correct_inputs = [t.matrix() for i in range(sequence_length)]
        noise_inputs = [t.matrix() for i in range(sequence_length)]
        learning_rate = t.scalar()

        stacked_correct_inputs = stack(correct_inputs)
        stacked_noise_inputs = stack(noise_inputs)

        correct_score, correct_prehidden = score(stacked_correct_inputs)
        noise_score, noise_prehidden = score(stacked_noise_inputs)
        unpenalized_loss = t.clip(1 - correct_score + noise_score, 0, 1e999)

        from hyperparameters import HYPERPARAMETERS
        if HYPERPARAMETERS["CW_EMBEDDING_L1_PENALTY"] != 0:
            l1penalty = t.sum(t.abs_(stacked_correct_inputs) + t.abs_(stacked_noise_inputs), axis=1).T * HYPERPARAMETERS["CW_EMBEDDING_L1_PENALTY"]
        else:
            l1penalty = t.as_tensor_variable(numpy.asarray(0, dtype=floatX))
#            l1penalty = t.as_tensor_variable(numpy.asarray((0,), dtype=floatX))
        loss = (unpenalized_loss.T + l1penalty).T

#        import sys
#        print >> sys.stderr, "FIXME: MODEL_LEARNING_RATE = fixed at 0.001"
#        MODEL_LEARNING_RATE = t.as_tensor_variable(numpy.asarray(0.001, dtype=floatX))

        total_loss = t.sum(loss)

        (dhidden_weights, dhidden_biases, doutput_weights, doutput_biases) = t.grad(total_loss, [hidden_weights, hidden_biases, output_weights, output_biases])
        dcorrect_inputs = t.grad(total_loss, correct_inputs)
        dnoise_inputs = t.grad(total_loss, noise_inputs)
        #print "REMOVEME", len(dcorrect_inputs)
        predict_inputs = correct_inputs
        train_inputs = correct_inputs + noise_inputs + [learning_rate]
        verbose_predict_inputs = predict_inputs
        predict_outputs = [correct_score]
        train_outputs = dcorrect_inputs + dnoise_inputs + [loss, unpenalized_loss, l1penalty, correct_score, noise_score]
        verbose_predict_outputs = [correct_score, correct_prehidden]

        import theano.gof.graph

        nnodes = len(theano.gof.graph.ops(predict_inputs, predict_outputs))
        print "About to compile predict function over %d ops [nodes]..." % nnodes
        predict_function = pfunc(predict_inputs, predict_outputs, mode=COMPILE_MODE)
        print "...done constructing graph for sequence_length=%d" % (sequence_length)

        nnodes = len(theano.gof.graph.ops(verbose_predict_inputs, verbose_predict_outputs))
        print "About to compile predict function over %d ops [nodes]..." % nnodes
        verbose_predict_function = pfunc(verbose_predict_inputs, verbose_predict_outputs, mode=COMPILE_MODE)
        print "...done constructing graph for sequence_length=%d" % (sequence_length)

        nnodes = len(theano.gof.graph.ops(train_inputs, train_outputs))
        print "About to compile train function over %d ops [nodes]..." % nnodes
        train_function = pfunc(train_inputs, train_outputs, mode=COMPILE_MODE, updates=[(p, p-learning_rate*gp) for p, gp in zip((hidden_weights, hidden_biases, output_weights, output_biases), (dhidden_weights, dhidden_biases, doutput_weights, doutput_biases))])
        print "...done constructing graph for sequence_length=%d" % (sequence_length)

        cached_functions[cachekey] = (predict_function, train_function, verbose_predict_function)
    return cached_functions[cachekey]

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
#    print "REMOVEME", correct_sequence
    r = fn(*(correct_sequence))
    assert len(r) == 1
    r = r[0]
    assert r.shape == (1, 1)
    return r[0,0]
def verbose_predict(correct_sequence, parameters):
    fn = functions(sequence_length=len(correct_sequence))[2]
    r = fn(*(correct_sequence))
    assert len(r) == 2
    (score, prehidden) = r
    assert score.shape == (1, 1)
    return score[0,0], prehidden
def train(correct_sequence, noise_sequence, learning_rate):
    assert len(correct_sequence) == len(noise_sequence)
    fn = functions(sequence_length=len(correct_sequence))[1]
    r = fn(*(correct_sequence + noise_sequence + [learning_rate]))
    dcorrect_inputs = r[:len(correct_sequence)]
    r = r[len(correct_sequence):]
    dnoise_inputs = r[:len(noise_sequence)]
    r = r[len(correct_sequence):]
#    print "REMOVEME", len(dcorrect_inputs), len(dnoise_inputs)
    (loss, unpenalized_loss, l1penalty, correct_score, noise_score) = r
#    if loss == 0:
#        for di in [dhidden_weights, dhidden_biases, doutput_weights, doutput_biases]:
#            assert (di == 0).all()

    return (dcorrect_inputs, dnoise_inputs, loss, unpenalized_loss, l1penalty, correct_score, noise_score)

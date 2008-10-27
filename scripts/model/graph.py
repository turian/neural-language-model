"""
@todo: "dot(one_hot(a,b), c) is probably equivalent to something like c[a,:]" -james
"""

import theano

from theano import tensor as t
from theano import scalar as s
from pylearn.onehotop import one_hot

from theano.tensor.basic import horizontal_stack
from theano.sandbox.nnet_ops import sigmoid, crossentropy_softmax_1hot, softmax
from theano.tensor import dot

from theano import gradient

import theano.compile
from miscglobals import LINKER, OPTIMIZER
mode = theano.compile.Mode(LINKER, OPTIMIZER)

import numpy

import hyperparameters

# TODO: Pure stochastic one-example-per training?
# TODO: Reuse subfunctions, e.g. autoassociators, instead of rewriting them
# TODO: DESCRIBEME: Better var names?

convolution_weights = t.dmatrix()
convolution_biases = t.dmatrix()

if hyperparameters.USE_SECOND_HIDDEN_LAYER == True:
    hidden2_weights = t.dmatrix()
    hidden2_biases = t.dmatrix()

unembedding_weights = t.dmatrix()
unembedding_biases = t.dmatrix()

# TODO: Include gradient steps in actual function, don't do them manually


class ScalarSoftsign(theano.scalar.UnaryScalarOp):
    @staticmethod
    def static_impl(x):
        return x / (1.0 + numpy.abs(x))
    def impl(self, x):
        return ScalarSoftsign.static_impl(x)
    def grad(self, (x, ), (gz, )):
        if 'float' in x.type.dtype:
            d = (1.0 + theano.scalar.abs_(x))
            return [gz / (d * d)]
        else:
            return NotImplemented
    def c_code(self, node, name, (x, ), (z, ), sub):
        if node.inputs[0].type in [theano.scalar.float32, theano.scalar.float64]:
            return "%(z)s = %(x)s / (1.0+fabs(%(x)s));" % locals()
        raise NotImplementedError('only floating point x is implemented')
scalar_softsign = ScalarSoftsign(theano.scalar.upgrade_to_float, name='scalar_softsign')
softsign = t.Elemwise(scalar_softsign, name='softsign')


def activation_function(r):
    if hyperparameters.ACTIVATION_FUNCTION == "sigmoid":
        return sigmoid(r)
    elif hyperparameters.ACTIVATION_FUNCTION == "tanh":
        return t.tanh(r)
    elif hyperparameters.ACTIVATION_FUNCTION == "softsign":
        return softsign(r)
    else:
        assert 0

def embed(tok):
    """
    @param p: t.dmatrix(), initial word embeddings (provided by Jason +
    Ronan) to be transformed into an initial representation.
    """
    return tok

def convolve(reprs):
    """
    Horizontally stack a list of representations, and then compress them
    one representation.
    @todo: Rename as 'convolute'?
    @todo: Horizontal stack of entire array at once?
    @todo: Ensure correct # of representations?
    """
    assert len(reprs) >= 2
    x = horizontal_stack(*reprs)
    return activation_function(dot(x, convolution_weights) + convolution_biases)

def hidden2_layer(repr):
    return activation_function(dot(repr, hidden2_weights) + hidden2_biases)

def output_layer(repr):
    return dot(repr, unembedding_weights) + unembedding_biases

def unembed(repr, p):
    """
    @param p: t.lvector(), vector of index of correct output
    """
    return crossentropy_softmax_1hot(output_layer(repr), p)

cached_functions = {}
def functions(sequence_length, convolution_width):
    """
    Return four Theano function that will:
    Take a window and convert to a representation,
    which is then unembedded (as a xent_softmax pair).
     * The first function does prediction.
     * The second function does prediction over all possible outputs.
     * The third function does validation (prediction and return of
     neuron outputs, to determine variances). Note that we only return
     the outputs of the *final* layer of neurons.
     * The fourth function does learning (it includes a gradient).

    @todo: Describe me better
    @todo: Create version that does not require target_output
    """
    p = (sequence_length, convolution_width)
    if p not in cached_functions:
        print "Need to construct graph for sequence_length=%d, convolution_width=%d..." % (sequence_length, convolution_width)
        # Create the sequence_length inputs.
        # Each is a t.dmatrix(), initial word embeddings (provided by
        # Jason + Ronan) to be transformed into an initial representation.
        # We could use a vector, but instead we use a matrix with one row.
        inputs = [t.dmatrix() for i in range(sequence_length)]

        target_output = t.lvector()

        # Embed each input
        embedded = [embed(i) for i in inputs]
        assert len(embedded) == convolution_width
        reprs = convolve(embedded)
        hidden1_layer_output = reprs

        if hyperparameters.USE_SECOND_HIDDEN_LAYER:
            reprs = hidden2_layer(reprs)
            hidden2_layer_output = reprs

        xent_softmax_output = unembed(reprs, target_output)
        crossentropy = xent_softmax_output[0]
        # The following line doesn't work because we have to extract the likelihood for each example
        #target_likelihood = xent_softmax_output[1][target_output]
        max_and_argmax = t.max_and_argmax(xent_softmax_output[1])
        argmax_class = max_and_argmax[1]
        max_pr = max_and_argmax[0]

        loss = t.sum(crossentropy)

        unembed_all = softmax(output_layer(reprs))

        if hyperparameters.USE_SECOND_HIDDEN_LAYER:
            (dconvolution_weights, dconvolution_biases, dhidden2_weights, dhidden2_biases, dunembedding_weights, dunembedding_biases) = t.grad(loss, [convolution_weights, convolution_biases, hidden2_weights, hidden2_biases, unembedding_weights, unembedding_biases])
            predict_all_inputs = inputs + [convolution_weights, convolution_biases, hidden2_weights, hidden2_biases, unembedding_weights, unembedding_biases]
            inputs = inputs + [target_output, convolution_weights, convolution_biases, hidden2_weights, hidden2_biases, unembedding_weights, unembedding_biases]
            learn_outputs = [loss, argmax_class, max_pr, dconvolution_weights, dconvolution_biases, dhidden2_weights, dhidden2_biases, dunembedding_weights, dunembedding_biases]
            validate_outputs = [loss, argmax_class, max_pr, hidden1_layer_output, hidden2_layer_output]
        else:
            (dconvolution_weights, dconvolution_biases, dunembedding_weights, dunembedding_biases) = t.grad(loss, [convolution_weights, convolution_biases, unembedding_weights, unembedding_biases])
            predict_all_inputs = inputs + [convolution_weights, convolution_biases, unembedding_weights, unembedding_biases]
            inputs = inputs + [target_output, convolution_weights, convolution_biases, unembedding_weights, unembedding_biases]
            learn_outputs = [loss, argmax_class, max_pr, dconvolution_weights, dconvolution_biases, dunembedding_weights, dunembedding_biases]
            validate_outputs = [loss, argmax_class, max_pr, hidden1_layer_output]

        predict_outputs = [loss, argmax_class, max_pr]
        predict_all_outputs = [unembed_all]

#        from breuleux import pprint
#        pp = pprint.pprinter()
#        print unicode(pp.process_graph(inputs, predict_outputs))

        import theano.gof.graph

        nnodes = len(theano.gof.graph.ops(inputs, predict_outputs))
        print "About to compile predict function over %d ops [nodes]..." % nnodes
        predict_function = theano.function(inputs, predict_outputs, mode=mode)
        print "...done constructing graph for sequence_length=%d, convolution_width=%d" % (sequence_length, convolution_width)

        nnodes = len(theano.gof.graph.ops(predict_all_inputs, predict_all_outputs))
        print "About to compile predict all function over %d ops [nodes]..." % nnodes
        predict_all_function = theano.function(predict_all_inputs, predict_all_outputs, mode=mode)
        print "...done constructing graph for sequence_length=%d, convolution_width=%d" % (sequence_length, convolution_width)

        nnodes = len(theano.gof.graph.ops(inputs, validate_outputs))
        print "About to compile validate function over %d ops [nodes]..." % nnodes
        validate_function = theano.function(inputs, validate_outputs, mode=mode)
        print "...done constructing graph for sequence_length=%d, convolution_width=%d" % (sequence_length, convolution_width)

        nnodes = len(theano.gof.graph.ops(inputs, learn_outputs))
        print "About to compile learn function over %d ops [nodes]..." % nnodes
        learn_function = theano.function(inputs, learn_outputs, mode=mode)
        print "...done constructing graph for sequence_length=%d, convolution_width=%d" % (sequence_length, convolution_width)

        cached_functions[p] = (predict_function, predict_all_function, validate_function, learn_function)
    return cached_functions[p]

def apply_function(fn, sequence, target_output, parameters):
    inputs = [numpy.asarray([token]) for token in sequence]
    if target_output != None:
        if hyperparameters.USE_SECOND_HIDDEN_LAYER:
            return fn(*(inputs + [numpy.asarray([target_output]), parameters.convolution_weights, parameters.convolution_biases, parameters.hidden2_weights, parameters.hidden2_biases, parameters.unembedding_weights, parameters.unembedding_biases]))
        else:
            return fn(*(inputs + [numpy.asarray([target_output]), parameters.convolution_weights, parameters.convolution_biases, parameters.unembedding_weights, parameters.unembedding_biases]))
    else:
        if hyperparameters.USE_SECOND_HIDDEN_LAYER:
            return fn(*(inputs + [parameters.convolution_weights, parameters.convolution_biases, parameters.hidden2_weights, parameters.hidden2_biases, parameters.unembedding_weights, parameters.unembedding_biases]))
        else:
            return fn(*(inputs + [parameters.convolution_weights, parameters.convolution_biases, parameters.unembedding_weights, parameters.unembedding_biases]))

def predict(sequence, target_output, parameters):
    fn = functions(sequence_length=len(sequence), convolution_width=parameters.convolution_width)[0]
    return apply_function(fn, sequence, target_output, parameters)

def predict_all(sequence, parameters):
    fn = functions(sequence_length=len(sequence), convolution_width=parameters.convolution_width)[1]
    return apply_function(fn, sequence, None, parameters)

def validate(sequence, target_output, parameters):
    fn = functions(sequence_length=len(sequence), convolution_width=parameters.convolution_width)[2]
    return apply_function(fn, sequence, target_output, parameters)

def learn(sequence, target_output, parameters):
    fn = functions(sequence_length=len(sequence), convolution_width=parameters.convolution_width)[3]
    return apply_function(fn, sequence, target_output, parameters)

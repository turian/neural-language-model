"""
@todo: WRITEME
"""

class Parameters:
    """
    Parameters used by the L{Model}.
    @todo: Document these
    """

    import hyperparameters
    import miscglobals
    import vocabularies
    import pylearn.sandbox.embeddings.parameters as embeddings
    def __init__(self, window_width=hyperparameters.TRAINING_WINDOW, input_dimension=None, embedding_size=hyperparameters.EMBEDDING_SIZE, output_vocabsize=vocabularies.labelmap.len, randomly_initialize=True, hidden_layers=None, seed=miscglobals.RANDOMSEED):
        """
        Initialize L{Model} parameters.

        Weights, if chosen randomly, are initialized to uniform value in
            [-hyperparameters.SCALE_INITIAL_WEIGHTS_BY / sqrt(number of inputs to this weights neuron), 
             +hyperparameters.SCALE_INITIAL_WEIGHTS_BY / sqrt(number of inputs to this weights neuron)].
        Biases are initialized to zero.
        
        @param randomly_initialize: If True, then randomly initialize
        according to the given seed. If False, then just use zeroes.

        @todo: Use L{hyperparameter.mode_weight_initialize}
        """
        if input_dimension == None:
            import hyperparameters, vocabularies
            import pylearn.sandbox.embeddings.parameters as embeddings
            # Set it to a default value
            if hyperparameters.USE_POS_TAG_FEATURES:
                input_dimension = embeddings.DIMENSIONS + vocabularies.tagmap.len
            else:
                input_dimension = embeddings.DIMENSIONS

        if hidden_layers == None:
            if hyperparameters.USE_SECOND_HIDDEN_LAYER:
                hidden_layers = 2
            else:
                hidden_layers = 1

        self.convolution_width      = convolution_width
        self.input_dimension        = input_dimension
        self.embedding_size    = embedding_size
        self.output_vocabsize       = output_vocabsize
        self.hidden_layers          = hidden_layers

        import numpy
        import hyperparameters
        if randomly_initialize:
            from pylearn.sandbox.weights import random_weights
            numpy.random.seed(seed)
            self.convolution_weights = random_weights(self.total_input_dimension,
                                                embedding_size, scale_by=hyperparameters.SCALE_INITIAL_WEIGHTS_BY)
            if self.hidden_layers == 2:
                self.hidden2_weights = random_weights(embedding_size, embedding_size, scale_by=hyperparameters.SCALE_INITIAL_WEIGHTS_BY)
            self.unembedding_weights = random_weights(embedding_size, output_vocabsize, scale_by=hyperparameters.SCALE_INITIAL_WEIGHTS_BY)
        else:
            self.convolution_weights = numpy.zeros((self.total_input_dimension, embedding_size))
            if self.hidden_layers == 2:
                self.hidden2_weights = numpy.zeros((embedding_size, embedding_size))
            self.unembedding_weights = numpy.zeros((embedding_size, output_vocabsize))

        self.convolution_biases = numpy.zeros((1, embedding_size))
        if self.hidden_layers == 2:
            self.hidden2_biases = numpy.zeros((1, embedding_size))
        self.unembedding_biases = numpy.zeros((1, output_vocabsize))

    total_input_dimension = property(lambda self: self.convolution_width * self.input_dimension)

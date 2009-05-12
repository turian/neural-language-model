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
    import vocabulary
    def __init__(self, window_size=hyperparameters.WINDOW_SIZE, vocab_size=vocabulary.wordmap.len, embedding_size=hyperparameters.EMBEDDING_SIZE, hidden_size=hyperparameters.HIDDEN_SIZE, seed=miscglobals.RANDOMSEED):
        """
        Initialize L{Model} parameters.
        """

        self.vocab_size     = vocab_size
        self.window_size    = window_size
        self.embedding_size = embedding_size
        self.hidden_size    = hidden_size
        self.output_size    = 1

        import numpy
        import hyperparameters

        from pylearn.algorithms.weights import random_weights
        numpy.random.seed(seed)
        self.embeddings = numpy.random.rand(self.vocab_size, hyperparameters.EMBEDDING_SIZE) * 2 - 1
        self.hidden_weights = random_weights(self.input_size, self.hidden_size, scale_by=hyperparameters.SCALE_INITIAL_WEIGHTS_BY)
        self.output_weights = random_weights(self.hidden_size, self.output_size, scale_by=hyperparameters.SCALE_INITIAL_WEIGHTS_BY)

        self.hidden_biases = numpy.zeros((1, self.hidden_size))
        self.output_biases = numpy.zeros((1, self.output_size))

    input_size = property(lambda self: self.window_size * self.embedding_size)
    
    def normalize_embeddings():
        """
        Normalize such that the l2 norm of every embedding is hyperparameters.EMBEDDING_SIZE
        @todo: l1 norm?
        """
        global embeddings
    
        l2norm = (embeddings * embeddings).sum(axis=1)
        l2norm = numpy.sqrt(l2norm.reshape((vocabsize, 1)))
    
        embeddings /= l2norm
        import math
        embeddings *= math.sqrt(hyperparameters.EMBEDDING_SIZE)
    
        # TODO: Assert that norm is correct
    #    l2norm = (embeddings * embeddings).sum(axis=1)
    #    print l2norm.shape
    #    print (l2norm == numpy.ones((vocabsize)) * hyperparameters.EMBEDDING_SIZE)
    #    print (l2norm == numpy.ones((vocabsize)) * hyperparameters.EMBEDDING_SIZE).all()

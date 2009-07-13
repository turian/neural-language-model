"""
@todo: WRITEME
"""

from hyperparameters import HYPERPARAMETERS

class Parameters:
    """
    Parameters used by the L{Model}.
    @todo: Document these
    """

    import hyperparameters
    import miscglobals
    import vocabulary
    def __init__(self, window_size=HYPERPARAMETERS["WINDOW_SIZE"], vocab_size=vocabulary.wordmap.len, embedding_size=HYPERPARAMETERS["EMBEDDING_SIZE"], hidden_size=HYPERPARAMETERS["HIDDEN_SIZE"], seed=miscglobals.RANDOMSEED):
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
        self.embeddings = numpy.random.rand(self.vocab_size, HYPERPARAMETERS["EMBEDDING_SIZE"]) * 2 - 1
        if HYPERPARAMETERS["NORMALIZE_EMBEDDINGS"]: self.normalize(range(self.vocab_size))
        self.hidden_weights = random_weights(self.input_size, self.hidden_size, scale_by=HYPERPARAMETERS["SCALE_INITIAL_WEIGHTS_BY"])
        self.output_weights = random_weights(self.hidden_size, self.output_size, scale_by=HYPERPARAMETERS["SCALE_INITIAL_WEIGHTS_BY"])

        self.hidden_biases = numpy.zeros((1, self.hidden_size))
        self.output_biases = numpy.zeros((1, self.output_size))

    input_size = property(lambda self: self.window_size * self.embedding_size)
    
    def normalize(self, indices):
        """
        Normalize such that the l2 norm of the embeddings indices passed in.
        @todo: l1 norm?
        @return: The normalized embeddings
        """
        import numpy
        l2norm = numpy.square(self.embeddings[indices]).sum(axis=1)
        l2norm = numpy.sqrt(l2norm.reshape((len(indices), 1)))

        self.embeddings[indices] /= l2norm
        import math
        self.embeddings[indices] *= math.sqrt(self.embeddings.shape[1])
    
        # TODO: Assert that norm is correct
    #    l2norm = (embeddings * embeddings).sum(axis=1)
    #    print l2norm.shape
    #    print (l2norm == numpy.ones((vocabsize)) * HYPERPARAMETERS["EMBEDDING_SIZE"])
    #    print (l2norm == numpy.ones((vocabsize)) * HYPERPARAMETERS["EMBEDDING_SIZE"]).all()

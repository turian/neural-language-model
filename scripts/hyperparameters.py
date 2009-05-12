from os.path import join

#: Not actually used directly, just for convenience
DATA_DIR = "/home/fringant2/lisa/turian/dev/python/language-model.predict-final-word/data/"

TRAINING_SENTENCES = join(DATA_DIR, "allwords.train.gz")
VOCABULARY = join(DATA_DIR, "allwords.vocabulary-5000.txt")

# Each embedded word representation has this width
EMBEDDING_SIZE = 50

# Predict with a window of five words at a time
WINDOW_SIZE = 5

HIDDEN_SIZE = 100

#: Scaling value to control range for weight initialization
#SCALE_INITIAL_WEIGHTS_BY = math.sqrt(3)
SCALE_INITIAL_WEIGHTS_BY = 1

# Which activation function to use?
#ACTIVATION_FUNCTION="sigmoid"
#ACTIVATION_FUNCTION="tanh"
ACTIVATION_FUNCTION="softsign"

LEARNING_RATE = 0.1

## number of (higher-order) quadratic filters for James's neuron
#NUMBER_OF_QUADRATIC_FILTERS=0
## We use this scaling factor for initial weights of quadratic filters,
## instead of SCALE_INITIAL_WEIGHTS_BY
## @note: Try between 10 and 0.01
#SCALE_QUADRATIC_INITIAL_WEIGHTS_BY = 1

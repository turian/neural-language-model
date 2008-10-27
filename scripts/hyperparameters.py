from os.path import join

#: Not actually used directly, just for convenience
DATA_DIR = "/home/fringant2/lisa/turian/dev/python/language-model.predict-middle-word/data/"

TRAINING_SENTENCES = join(DATA_DIR, "allwords.gz")
VOCABULARY = join(DATA_DIR, "allwords.vocabulary-200.txt")

# Each embedded word representation has this width
EMBEDDING_SIZE = 20

# Train with a window of five words at a time
TRAINING_WINDOW = 5

## Use a second hidden layer?
## If so, it will have REPRESENTATION_SIZE units.
## @todo: This is brittle, and so is the implementation!
#USE_SECOND_HIDDEN_LAYER = False
#
##: Ronan's trick: Divide learning rate for all connections into a neuron
## by the number of inputs into that neuron.
## Note: Not clear what to do when the weight is shared. Maybe divide by
## sqrt of number of times it is used?
#DIVIDE_LEARNING_RATE_BY_NUMBER_OF_INPUTS = True
#
##: Scaling value to control range for weight initialization
##SCALE_INITIAL_WEIGHTS_BY = math.sqrt(3)
#SCALE_INITIAL_WEIGHTS_BY = 1
#
## Which activation function to use?
##ACTIVATION_FUNCTION="sigmoid"
#ACTIVATION_FUNCTION="tanh"
##ACTIVATION_FUNCTION="softsign"
#
## number of (higher-order) quadratic filters for James's neuron
#NUMBER_OF_QUADRATIC_FILTERS=0
## We use this scaling factor for initial weights of quadratic filters,
## instead of SCALE_INITIAL_WEIGHTS_BY
## @note: Try between 10 and 0.01
#SCALE_QUADRATIC_INITIAL_WEIGHTS_BY = 1

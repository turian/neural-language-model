from os.path import join, expanduser

#: Not actually used directly, just for convenience
DATA_DIR = join(expanduser("~"), "dev/python/language-model.predict-final-word/data/")

#TRAIN_SENTENCES = join(DATA_DIR, "allwords.train.gz")
##VALIDATION_SENTENCES = join(DATA_DIR, "allwords.validation.gz")
#VALIDATION_SENTENCES = join(DATA_DIR, "allwords.validation-100.gz")
#VOCABULARY = join(DATA_DIR, "allwords.vocabulary-5000.txt")

TRAIN_SENTENCES = join(DATA_DIR, "wikitext.train.txt.gz")
#VALIDATION_SENTENCES = join(DATA_DIR, "wikitext.validation.txt")
VALIDATION_SENTENCES = join(DATA_DIR, "wikitext.validation-200.txt")
#VALIDATION_SENTENCES = join(DATA_DIR, "wikitext.validation-100.txt")
#VOCABULARY = join(DATA_DIR, "vocabulary-wikitext-10000.txt.gz")
VOCABULARY = {5000: join(DATA_DIR, "vocabulary-wikitext-5000.txt.gz"),
10000: join(DATA_DIR, "vocabulary-wikitext-10000.txt.gz"),
20000: join(DATA_DIR, "vocabulary-wikitext-20000.txt.gz")}

#NORMALIZE_EMBEDDINGS = True
NORMALIZE_EMBEDDINGS = False
#UPDATES_PER_NORMALIZE_EMBEDDINGS = 1000

#VOCABULARY_SIZE = 5000
#VOCABULARY_SIZE = 10000
VOCABULARY_SIZE = 20000

NGRAM_FOR_TRAINING_NOISE = 1

NGRAMS = {(1, 5000): join(DATA_DIR, "1grams-wikitext-5000.json.gz"),
(1, 10000): join(DATA_DIR, "1grams-wikitext-10000.json.gz"),
(1, 20000): join(DATA_DIR, "1grams-wikitext-20000.json.gz")}

# Number of instances of each ngram to add, for smoothing.
TRAINING_NOISE_SMOOTHING_ADDITION = 0

# Each embedded word representation has this width
EMBEDDING_SIZE = 50
#EMBEDDING_SIZE = 5

# Predict with a window of five words at a time
WINDOW_SIZE = 5

HIDDEN_SIZE = 100
#HIDDEN_SIZE = 10

#: Scaling value to control range for weight initialization
#SCALE_INITIAL_WEIGHTS_BY = math.sqrt(3)
SCALE_INITIAL_WEIGHTS_BY = 1

# Which activation function to use?
#ACTIVATION_FUNCTION="sigmoid"
#ACTIVATION_FUNCTION="tanh"
ACTIVATION_FUNCTION="softsign"

#LEARNING_RATE = 0.001
#LEARNING_RATE = 0.01
LEARNING_RATE = 0.1

# The "fan-in" of the embeddings. We could make this 1 or EMBEDDING_SIZE or EMBEDDING_SIZE*WINDOW_SIZE
FAN_IN_OF_EMBEDDINGS = 1
# Should we divide the learning rate by the fan-in?
DIVIDE_LEARNING_RATE_BY_FANIN = False
# Should we divide the learning rate by sqrt(fan-in)?
DIVIDE_LEARNING_RATE_BY_SQRT_FANIN = True

## number of (higher-order) quadratic filters for James's neuron
#NUMBER_OF_QUADRATIC_FILTERS=0
## We use this scaling factor for initial weights of quadratic filters,
## instead of SCALE_INITIAL_WEIGHTS_BY
## @note: Try between 10 and 0.01
#SCALE_QUADRATIC_INITIAL_WEIGHTS_BY = 1

# Validate after this many examples
VALIDATE_EVERY = 1000000
#VALIDATE_EVERY = 1000

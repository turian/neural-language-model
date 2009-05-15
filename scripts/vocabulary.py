import hyperparameters
import common.featuremap as featuremap
wordmap = featuremap.get(name="words-%d" % (hyperparameters.VOCABULARY_SIZE), synchronize=True)

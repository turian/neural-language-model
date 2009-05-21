import hyperparameters
import common.featuremap as featuremap
wordmap = featuremap.get(name="words-%d" % (hyperparameters.VOCABULARY_SIZE), synchronize=True)

def read():
    for l in myopen(hyperparameters.VOCABULARY[hyperparameters.VOCABULARY_SIZE]):
        (cnt, word) = string.split(l)
        wordmap.id(word, can_add=True)
    wordmap.dump()
    wordmap.readonly = True

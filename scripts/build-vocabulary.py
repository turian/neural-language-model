#!/usr/bin/python

import vocabulary

if __name__ == "__main__":
    import hyperparameters
    import common.options
    import common.file
    hyperparameters.__dict__.update(common.options.reparse(hyperparameters.__dict__))

    import string
    for l in common.file.myopen(hyperparameters.VOCABULARY[hyperparameters.VOCABULARY_SIZE]):
        print string.split(l)
        (cnt, w) = string.split(l)
        vocabulary.wordmap.id(w, can_add=True)
    vocabulary.wordmap.dump()

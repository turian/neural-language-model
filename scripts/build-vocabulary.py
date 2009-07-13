#!/usr/bin/python

import vocabulary
import common.idmap

if __name__ == "__main__":
    import hyperparameters
    import common.options
    import common.file
    hyperparameters.__dict__.update(common.options.reparse(hyperparameters.__dict__)[0])

    words = set()

    import string
    for l in common.file.myopen(hyperparameters.VOCABULARY):
        (cnt, w) = string.split(l)
        words.add(w)

    vocabulary.write(common.idmap.IDmap(words, allow_unknown=hyperparameters.INCLUDE_UNKNOWN_WORD))

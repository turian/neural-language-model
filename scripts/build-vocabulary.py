#!/usr/bin/python

import vocabulary
import common.idmap

if __name__ == "__main__":
    import hyperparameters
    import common.options
    import common.file
    hyperparameters.__dict__.update(common.options.reparse(hyperparameters.__dict__)[0])

    words = []

    import string
    for i, l in enumerate(common.file.myopen(hyperparameters.VOCABULARY)):
        if hyperparameters.INCLUDE_UNKNOWN_WORD and i+1 >= hyperparameters.VOCABULARY_SIZE:
            break
        if not hyperparameters.INCLUDE_UNKNOWN_WORD and i >= hyperparameters.VOCABULARY_SIZE:
            break
        (cnt, w) = string.split(l)
        words.append(w)

    v = common.idmap.IDmap(words, allow_unknown=hyperparameters.INCLUDE_UNKNOWN_WORD)
    assert v.len == hyperparameters.VOCABULARY_SIZE
    vocabulary.write(common.idmap.IDmap(words, allow_unknown=hyperparameters.INCLUDE_UNKNOWN_WORD))

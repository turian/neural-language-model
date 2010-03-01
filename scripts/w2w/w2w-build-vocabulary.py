#!/usr/bin/env python

if __name__ == "__main__":
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    HYPERPARAMETERS, options, args, newkeystr = common.options.reparse(HYPERPARAMETERS)
    import hyperparameters

    import w2w.corpora

#    import vocabulary
#    import common.idmap
#
#    words = []
#
#    import string
#    for i, l in enumerate(common.file.myopen(HYPERPARAMETERS["MONOLINGUAL_VOCABULARY"])):
#        if HYPERPARAMETERS["INCLUDE_UNKNOWN_WORD"] and i+1 >= HYPERPARAMETERS["MONOLINGUAL_VOCABULARY_SIZE"]:
#            break
#        if not HYPERPARAMETERS["INCLUDE_UNKNOWN_WORD"] and i >= HYPERPARAMETERS["MONOLINGUAL_VOCABULARY_SIZE"]:
#            break
#        (cnt, w) = string.split(l)
#        words.append(w)
#
#    v = common.idmap.IDmap(words, allow_unknown=HYPERPARAMETERS["INCLUDE_UNKNOWN_WORD"])
#    assert v.len == HYPERPARAMETERS["MONOLINGUAL_VOCABULARY_SIZE"]
#    vocabulary.write(v)

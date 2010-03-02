#!/usr/bin/env python
"""
Dump the w2w vocaulary.
"""

if __name__ == "__main__":
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    HYPERPARAMETERS, options, args, newkeystr = common.options.reparse(HYPERPARAMETERS)
    import hyperparameters

    from vocabulary import wordmap
    for w in wordmap().all:
        print w

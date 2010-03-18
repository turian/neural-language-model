#!/usr/bin/env python
"""
Dump the w2w target vocabulary.
"""

import sys

if __name__ == "__main__":
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    HYPERPARAMETERS, options, args, newkeystr = common.options.reparse(HYPERPARAMETERS)

    import logging
    logging.basicConfig(level=logging.INFO)

    import w2w.examples
    for e in w2w.examples.get_all_training_examples_cached():
        print e

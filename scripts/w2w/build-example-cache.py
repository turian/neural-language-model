#!/usr/bin/env python
"""
Extract all training examples, and cache them.
"""

if __name__ == "__main__":
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    HYPERPARAMETERS, options, args, newkeystr = common.options.reparse(HYPERPARAMETERS)
    import hyperparameters

    import logging
    logging.basicConfig(level=logging.INFO)

    import w2w.examples
    w2w.examples.all_training_examples_cached()

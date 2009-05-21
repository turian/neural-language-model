#!/usr/bin/env python
"""
Dump n-gram counts over entire training data as YAML.
"""

import sys
from common.stats import stats

from collections import defaultdict
cnt = defaultdict(int)
if __name__ == "__main__":
    import hyperparameters
    import common.options
    hyperparameters.__dict__.update(common.options.reparse(hyperparameters.__dict__))

    import vocabulary
    print "Reading vocab"
    vocabulary.read()
    from vocabulary import wordmap

    import train
    for (i, e) in enumerate(train.get_train_example()):
        cnt[tuple([wordmap.str(t) for t in e])] += 1
        if i % 10000 == 0:
            print >> sys.stderr, "Read %d examples" % i
            print >> sys.stderr, stats()
    import common.myyaml
    print common.myyaml.dump(cnt)

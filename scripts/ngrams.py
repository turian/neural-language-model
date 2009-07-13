#!/usr/bin/env python
"""
Dump n-gram counts over entire training data as YAML.
"""

import sys
from common.stats import stats

from collections import defaultdict
cnt = defaultdict(int)
if __name__ == "__main__":
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    HYPERPARAMETERS, options, args = common.options.reparse(HYPERPARAMETERS)
    import hyperparameters

    import vocabulary
    print >> sys.stderr, "Reading vocab"
    vocabulary.read()
    from vocabulary import wordmap

    import train
    for (i, e) in enumerate(train.get_train_example()):
        cnt[tuple([wordmap.str(t) for t in e])] += 1
        if i % 10000 == 0:
            print >> sys.stderr, "Read %d examples" % i
            print >> sys.stderr, stats()
        if i > 100000000:
            break
    cnt = [(t, cnt[t]) for t in cnt]
    import common.json
    common.json.dump(cnt, sys.stdout)

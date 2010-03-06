#!/usr/bin/python
"""
For the N files given as command line arguments, filter the sentences
to be only those in which the first file contains a word that lemmatizes
to one of the W2W FOCUS LEMMAS.
We write files that are prefixed by "filtered-"
"""

from common.str import percent
import string
import sys

import common.hyperparameters, common.options
HYPERPARAMETERS = common.hyperparameters.read("language-model")
HYPERPARAMETERS, options, args, newkeystr = common.options.reparse(HYPERPARAMETERS)

if HYPERPARAMETERS["W2W FOCUS LEMMAS"] is None or len (HYPERPARAMETERS["W2W FOCUS LEMMAS"]) == 0:
    print >> sys.stderr, "There are no focus lemmas, hence we have nothing to filter"
    sys.exit(0)

assert len(args) >= 1

from common.stats import stats
from lemmatizer import lemmatize

print >> sys.stderr, "Loaded Morphological analyizer"
print >> sys.stderr, stats()

from itertools import izip
import os.path, os

filenames = args
outfilenames = [os.path.join(os.path.dirname(f), "filtered-%s" % os.path.basename(f)) for f in filenames]

print >> sys.stderr, "Reading from %s" % `filenames`
print >> sys.stderr, "Writing to %s" % `outfilenames`

for f in filenames: assert os.path.exists(f)
for f in outfilenames:
    if os.path.exists(f):
        print >> sys.stderr, "Warning, going to overwrite %s" % f

#print "Sleeping for 10 seconds..."
#import time
#time.sleep(10)

inf = [open(f) for f in filenames]
outf = [open(f, "wt") for f in outfilenames]

tot = 0
cnt = 0
for lines in izip(*inf):
    tot += 1
    keep = False
    for w in string.split(lines[0]):
        if lemmatize("en", w) in HYPERPARAMETERS["W2W FOCUS LEMMAS"]:
            keep = True
            break
    if keep:
        cnt += 1
        for l, f in izip(lines, outf):
            f.write(l)
    if tot % 10000 == 0:
        print >> sys.stderr, "%s lines kept" % percent(cnt, tot)
        print >> sys.stderr, stats()

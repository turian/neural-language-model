#!/usr/bin/env python
#
#  Plot a histogram of the absolute values of model embeddings
#
#

PERCENT = 0.01
import random

import sys
import matplotlib
matplotlib.use( 'Agg' ) # Use non-GUI backend
import pylab

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-m", "--modelfile", dest="modelfile")
(options, args) = parser.parse_args()
assert options.modelfile is not None

histfile = "%s.weight-histogram.png" % options.modelfile

import cPickle
m = cPickle.load(open(options.modelfile))
#print m.parameters.embeddings.shape

values = []

from vocabulary import wordmap
for i in range(m.parameters.vocab_size):
    for v in m.parameters.embeddings[i]:
        if random.random() < PERCENT:
            values.append(abs(v))
values.sort()

print >> sys.stderr, "%d values read (at %f percent) of %d embeddings, %d/%f/%d = %f" % (len(values), PERCENT, m.parameters.vocab_size, len(values), PERCENT, m.parameters.vocab_size, len(values)/PERCENT/m.parameters.vocab_size)

x = []
for i, v in enumerate(values):
    x.append(1./(len(values)-1) * i)

print >> sys.stderr, 'Writing weight histogram to %s' % histfile

pylab.plot(x, values)
pylab.savefig(histfile)
pylab.show()

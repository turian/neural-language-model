#!/usr/bin/env python
#
#  Plot a histogram of the absolute values of model embeddings
#
#

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
        values.append(abs(v))
values.sort()

x = []
for i, v in enumerate(values):
    x.append(1./(len(values)-1) * i)

print >> sys.stderr, 'Writing weight histogram to %s' % histfile

pylab.plot(x, values)
pylab.savefig(histfile)
pylab.show()

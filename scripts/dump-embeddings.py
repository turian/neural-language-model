#!/usr/bin/python

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-m", "--modelfile", dest="modelfile")
(options, args) = parser.parse_args()
assert options.modelfile is not None

import cPickle
m = cPickle.load(open(options.modelfile))
#print m.parameters.embeddings.shape

from vocabulary import wordmap
for i in range(m.parameters.vocab_size):
    print wordmap.str(i),
    for v in m.parameters.embeddings[i]:
        print v,
    print

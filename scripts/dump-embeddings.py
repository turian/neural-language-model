#!/usr/bin/python

modelfile = "LOGS.NOBACKUP/run-058029f096a276b903460a625999635cc582d019d00572c5aa42f5ca/model-380000001.pkl"

import cPickle
m = cPickle.load(open(modelfile))
#print m.parameters.embeddings.shape

from vocabulary import wordmap
for i in range(m.parameters.vocab_size):
    print wordmap.str(i),
    for v in m.parameters.embeddings[i]:
        print v,
    print

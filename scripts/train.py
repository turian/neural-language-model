#!/usr/bin/python

import common.dump
import hyperparameters, miscglobals
common.dump.vars(hyperparameters)
common.dump.vars(miscglobals)

import random, numpy
random.seed(miscglobals.RANDOMSEED)
numpy.random.seed(miscglobals.RANDOMSEED)

from common.file import myopen

from vocabulary import wordmap

import string

vocabsize = None
def read_vocabulary():
    global vocabsize
    for l in myopen(hyperparameters.VOCABULARY):
        (cnt, word) = string.split(l)
        wordmap.id(word, can_add=True)
    wordmap.dump()
    wordmap.readonly = True
    vocabsize = wordmap.len

def get_training_example():
    prevwords = []
    for w in myopen(hyperparameters.TRAINING_SENTENCES):
        w = string.strip(w)
        if wordmap.exists(w):
            prevwords.append(wordmap.id(w))
            if len(prevwords) > hyperparameters.WINDOW_SIZE:
                yield prevwords[-hyperparameters.WINDOW_SIZE:]
        else:
            prevwords = []

print "Reading vocab"
read_vocabulary()
#print "Reading examples"
#print len(read_training_examples())

def corrupt_example(e):
    import copy
    e = copy.copy(e)
    last = e[-1]
    while e[-1] == last:
        e[-1] = random.randint(0, vocabsize-1)
    return e

import model
m = model.Model()
for e in get_training_example():
    ecorrupt = corrupt_example(e)
    print e, ecorrupt
    print m.predict(e)

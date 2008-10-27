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
            if len(prevwords) > hyperparameters.TRAINING_WINDOW:
                yield prevwords[-hyperparameters.TRAINING_WINDOW:]
        else:
            prevwords = []

embeddings = None
def initialize_embeddings():
    global embeddings
    embeddings = numpy.random.rand(vocabsize, hyperparameters.EMBEDDING_SIZE) * 2 - 1

import math
def normalize_embeddings():
    """
    Normalize such that the l2 norm of every embedding is hyperparameters.EMBEDDING_SIZE
    @todo: l1 norm?
    """
    global embeddings

    l2norm = (embeddings * embeddings).sum(axis=1)
    l2norm = numpy.sqrt(l2norm.reshape((200, 1)))

    embeddings /= l2norm
    embeddings *= math.sqrt(hyperparameters.EMBEDDING_SIZE)

    # TODO: Assert that norm is correct
#    l2norm = (embeddings * embeddings).sum(axis=1)
#    print l2norm.shape
#    print (l2norm == numpy.ones((vocabsize)) * hyperparameters.EMBEDDING_SIZE)
#    print (l2norm == numpy.ones((vocabsize)) * hyperparameters.EMBEDDING_SIZE).all()

nearest_neighbors = []
def recompute_nearest_neighbors():
    """
    @todo: Try other distance measures?
    @todo: Other number of nearest neighbors (not 10)?
    """
    NEIGHBORS = 10
    global nearest_neighbors
    nearest_neighbors = [[] for i in range(vocabsize)]
    for i in range(vocabsize):
        nn = []
        for j in range(vocabsize):
            if i == j: continue
            d = (embeddings[i] - embeddings[j])
            dist = (d * d).sum()
            nn.append((dist, j)) 
        nn.sort()
        nearest_neighbors[i] = [j for (dist, j) in nn[:NEIGHBORS]]

print "Reading vocab"
read_vocabulary()
#print "Reading examples"
#print len(read_training_examples())

initialize_embeddings()
normalize_embeddings()
recompute_nearest_neighbors()

for e in get_training_example():
    print e

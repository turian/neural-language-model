#!/usr/bin/python

import common.dump
import hyperparameters, miscglobals
common.dump.vars(hyperparameters)
common.dump.vars(miscglobals)

from common.stats import stats

import sys

import random, numpy
random.seed(miscglobals.RANDOMSEED)
numpy.random.seed(miscglobals.RANDOMSEED)

from common.file import myopen

from vocabulary import wordmap

import string

def read_vocabulary():
    for l in myopen(hyperparameters.VOCABULARY):
        (cnt, word) = string.split(l)
        wordmap.id(word, can_add=True)
    wordmap.dump()
    wordmap.readonly = True

def get_train_example():
    prevwords = []
    for w in myopen(hyperparameters.TRAIN_SENTENCES):
        w = string.strip(w)
        if wordmap.exists(w):
            prevwords.append(wordmap.id(w))
            if len(prevwords) > hyperparameters.WINDOW_SIZE:
                yield prevwords[-hyperparameters.WINDOW_SIZE:]
        else:
            prevwords = []

def get_validation_example():
    prevwords = []
    for w in myopen(hyperparameters.VALIDATION_SENTENCES):
        w = string.strip(w)
        if wordmap.exists(w):
            prevwords.append(wordmap.id(w))
            if len(prevwords) > hyperparameters.WINDOW_SIZE:
                yield prevwords[-hyperparameters.WINDOW_SIZE:]
        else:
            prevwords = []

print "Reading vocab"
read_vocabulary()

def validate(cnt):
    import math
    logranks = []
    print >> sys.stderr, "BEGINNING VALIDATION AT TRAINING STEP %d" % cnt
    print >> sys.stderr, stats()
    for (i, ve) in enumerate(get_validation_example()):
        logranks.append(math.log(m.validate(ve)))
        if (i+1) % 10 == 0:
            print >> sys.stderr, "Training step %d, validating example %d, mean(logrank) = %.2f, stddev(logrank) = %.2f" % (cnt, i+1, numpy.mean(numpy.array(logranks)), numpy.std(numpy.array(logranks)))
            print >> sys.stderr, stats()
    print >> sys.stderr, "FINAL VALIDATION AT TRAINING STEP %d: mean(logrank) = %.2f, stddev(logrank) = %.2f" % (cnt, numpy.mean(numpy.array(logranks)), numpy.std(numpy.array(logranks)))
    print >> sys.stderr, stats()

import model
m = model.Model()
for (cnt, e) in enumerate(get_train_example()):
    m.train(e)

    if (cnt+1) % 100 == 0:
        print >> sys.stderr, "Finished training step %d" % cnt
    if (cnt+1) % hyperparameters.VALIDATE_EVERY == 0:
        validate(cnt+1)

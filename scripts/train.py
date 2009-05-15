#!/usr/bin/python

import common.dump
import hyperparameters, miscglobals

import common.options
hyperparameters.__dict__.update(common.options.reparse(hyperparameters.__dict__))
#miscglobals.__dict__ = common.options.reparse(miscglobals.__dict__)

from common import myyaml
import sys
print >> sys.stderr, myyaml.dump(common.dump.vars_seq([hyperparameters, miscglobals]))

from common.stats import stats

import sys

import random, numpy
random.seed(miscglobals.RANDOMSEED)
numpy.random.seed(miscglobals.RANDOMSEED)

from common.file import myopen

from vocabulary import wordmap

import string

def read_vocabulary():
    for l in myopen(hyperparameters.VOCABULARY[hyperparameters.VOCABULARY_SIZE]):
        (cnt, word) = string.split(l)
        wordmap.id(word, can_add=True)
    wordmap.dump()
    wordmap.readonly = True

def get_train_example():
    for l in myopen(hyperparameters.TRAIN_SENTENCES):
        prevwords = []
        for w in string.split(l):
            w = string.strip(w)
            if wordmap.exists(w):
                prevwords.append(wordmap.id(w))
                if len(prevwords) >= hyperparameters.WINDOW_SIZE:
                    yield prevwords[-hyperparameters.WINDOW_SIZE:]
            else:
                prevwords = []

def get_validation_example():
    for l in myopen(hyperparameters.VALIDATION_SENTENCES):
        prevwords = []
        for w in string.split(l):
            w = string.strip(w)
            if wordmap.exists(w):
                prevwords.append(wordmap.id(w))
                if len(prevwords) >= hyperparameters.WINDOW_SIZE:
                    yield prevwords[-hyperparameters.WINDOW_SIZE:]
            else:
                prevwords = []

#ves = [e for e in get_validation_example()]
#import random
#random.shuffle(ves)
#for e in ves[:1000]:
#    print string.join([wordmap.str(id) for id in e])

print "Reading vocab"
read_vocabulary()

def validate(cnt):
    import math
    logranks = []
    print >> sys.stderr, "BEGINNING VALIDATION AT TRAINING STEP %d" % cnt
    print >> sys.stderr, stats()
    i = 0
    for (i, ve) in enumerate(get_validation_example()):
#        print >> sys.stderr, [wordmap.str(id) for id in ve]
        logranks.append(math.log(m.validate(ve)))
        if (i+1) % 10 == 0:
            print >> sys.stderr, "Training step %d, validating example %d, mean(logrank) = %.2f, stddev(logrank) = %.2f" % (cnt, i+1, numpy.mean(numpy.array(logranks)), numpy.std(numpy.array(logranks)))
            print >> sys.stderr, stats()
    print >> sys.stderr, "FINAL VALIDATION AT TRAINING STEP %d: mean(logrank) = %.2f, stddev(logrank) = %.2f, cnt = %d" % (cnt, numpy.mean(numpy.array(logranks)), numpy.std(numpy.array(logranks)), i+1)
    print >> sys.stderr, stats()

import model
m = model.Model()
#validate(0)
for (cnt, e) in enumerate(get_train_example()):
#    print [wordmap.str(id) for id in e]
    m.train(e)

    if (cnt+1) % 100 == 0:
        print >> sys.stderr, "Finished training step %d" % (cnt+1)
    if (cnt+1) % 10000 == 0:
        print >> sys.stderr, stats()
    if (cnt+1) % hyperparameters.VALIDATE_EVERY == 0:
        validate(cnt+1)

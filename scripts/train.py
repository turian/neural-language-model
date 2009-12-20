#!/usr/bin/env python

import sys
import string
import common.dump
from common.file import myopen
from common.stats import stats

import miscglobals
import logging

import examples
import verbosedebug

def validate(cnt):
    import math
    logranks = []
    logging.info("BEGINNING VALIDATION AT TRAINING STEP %d" % cnt)
    logging.info(stats())
    i = 0
    for (i, ve) in enumerate(examples.get_validation_example()):
#        logging.info([wordmap.str(id) for id in ve])
        logranks.append(math.log(m.validate(ve)))
        if (i+1) % 10 == 0:
            logging.info("Training step %d, validating example %d, mean(logrank) = %.2f, stddev(logrank) = %.2f" % (cnt, i+1, numpy.mean(numpy.array(logranks)), numpy.std(numpy.array(logranks))))
            logging.info(stats())
    logging.info("FINAL VALIDATION AT TRAINING STEP %d: mean(logrank) = %.2f, stddev(logrank) = %.2f, cnt = %d" % (cnt, numpy.mean(numpy.array(logranks)), numpy.std(numpy.array(logranks)), i+1))
    logging.info(stats())
#    print "FINAL VALIDATION AT TRAINING STEP %d: mean(logrank) = %.2f, stddev(logrank) = %.2f, cnt = %d" % (cnt, numpy.mean(numpy.array(logranks)), numpy.std(numpy.array(logranks)), i+1)
#    print stats()

def save_state(m, cnt):
    import os.path
    filename = os.path.join(rundir, "model-%d.pkl" % cnt)
    logging.info("Writing model to %s..." % filename)
    logging.info(stats())
    import cPickle
    cPickle.dump(m, myopen(filename, "wb"), protocol=-1)
    logging.info("...done writing model to %s" % filename)
    logging.info(stats())

if __name__ == "__main__":
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    HYPERPARAMETERS, options, args, newkeystr = common.options.reparse(HYPERPARAMETERS)
    import hyperparameters

    from common import myyaml
    import sys
    print >> sys.stderr, myyaml.dump(common.dump.vars_seq([hyperparameters, miscglobals]))

    import noise
    indexed_weights = noise.indexed_weights()

    rundir = common.dump.create_canonical_directory(HYPERPARAMETERS)

    import os.path, os
    logfile = os.path.join(rundir, "log")
    if newkeystr != "":
        verboselogfile = os.path.join(rundir, "log%s" % newkeystr)
        print >> sys.stderr, "Logging to %s, and creating link %s" % (logfile, verboselogfile)
        os.system("ln -s log %s " % (verboselogfile))
    else:
        print >> sys.stderr, "Logging to %s, not creating any link because of default settings" % logfile
    #logging.basicConfig(filename=logfile,level=logging.DEBUG)
    logging.basicConfig(filename=logfile, filemode="w", level=logging.DEBUG)
    logging.info(myyaml.dump(common.dump.vars_seq([hyperparameters, miscglobals])))

    import random, numpy
    random.seed(miscglobals.RANDOMSEED)
    numpy.random.seed(miscglobals.RANDOMSEED)

    import vocabulary
#    logging.info("Reading vocab")
#    vocabulary.read()
    
    import model
    m = model.Model()
    #validate(0)
    epoch = 0
    cnt = 0
    verbosedebug.verbosedebug(cnt, m)
    while 1:
        epoch += 1
        logging.info("STARTING EPOCH #%d" % epoch)
        for e in examples.get_train_example():
            cnt += 1
        #    print [wordmap.str(id) for id in e]
            m.train(e)

            #validate(cnt)
            if cnt % 1000 == 0:
                logging.info("Finished training step %d (epoch %d)" % (cnt, epoch))
#                print ("Finished training step %d (epoch %d)" % (cnt, epoch))
            if cnt % 10000 == 0:
                verbosedebug.verbosedebug(cnt, m)
                if os.path.exists(os.path.join(rundir, "BAD")):
                    logging.info("Detected file: %s\nSTOPPING" % os.path.join(rundir, "BAD"))
                    sys.stderr.write("Detected file: %s\nSTOPPING\n" % os.path.join(rundir, "BAD"))
                    sys.exit(0)
            if cnt % HYPERPARAMETERS["VALIDATE_EVERY"] == 0:
                save_state(m, cnt)
                verbosedebug.visualize(cnt, m, rundir, randomized=False)
                verbosedebug.visualize(cnt, m, rundir, randomized=True)
                validate(cnt)

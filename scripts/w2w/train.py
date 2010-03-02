#!/usr/bin/env python

import sys
import string
import common.dump
from common.file import myopen
from common.stats import stats

import miscglobals
import logging

import w2w.examples
#import diagnostics
#import state

#def validate(cnt):
#    import math
#    logranks = []
#    logging.info("BEGINNING VALIDATION AT TRAINING STEP %d" % cnt)
#    logging.info(stats())
#    i = 0
#    for (i, ve) in enumerate(examples.get_validation_example()):
##        logging.info([wordmap.str(id) for id in ve])
#        logranks.append(math.log(m.validate(ve)))
#        if (i+1) % 10 == 0:
#            logging.info("Training step %d, validating example %d, mean(logrank) = %.2f, stddev(logrank) = %.2f" % (cnt, i+1, numpy.mean(numpy.array(logranks)), numpy.std(numpy.array(logranks))))
#            logging.info(stats())
#    logging.info("FINAL VALIDATION AT TRAINING STEP %d: mean(logrank) = %.2f, stddev(logrank) = %.2f, cnt = %d" % (cnt, numpy.mean(numpy.array(logranks)), numpy.std(numpy.array(logranks)), i+1))
#    logging.info(stats())
##    print "FINAL VALIDATION AT TRAINING STEP %d: mean(logrank) = %.2f, stddev(logrank) = %.2f, cnt = %d" % (cnt, numpy.mean(numpy.array(logranks)), numpy.std(numpy.array(logranks)), i+1)
##    print stats()

if __name__ == "__main__":
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("language-model")
    HYPERPARAMETERS, options, args, newkeystr = common.options.reparse(HYPERPARAMETERS)
    import hyperparameters

    from common import myyaml
    import sys
    print >> sys.stderr, myyaml.dump(common.dump.vars_seq([hyperparameters, miscglobals]))

    # We do not allow sophisticated training noise
    assert HYPERPARAMETERS["NGRAM_FOR_TRAINING_NOISE"] == 0

    from rundir import rundir
    rundir = rundir()

    import os.path, os
    logfile = os.path.join(rundir, "log")
    if newkeystr != "":
        verboselogfile = os.path.join(rundir, "log%s" % newkeystr)
        print >> sys.stderr, "Logging to %s, and creating link %s" % (logfile, verboselogfile)
        os.system("ln -s log %s " % (verboselogfile))
    else:
        print >> sys.stderr, "Logging to %s, not creating any link because of default settings" % logfile

    import random, numpy
    random.seed(miscglobals.RANDOMSEED)
    numpy.random.seed(miscglobals.RANDOMSEED)

#    import vocabulary
##    logging.info("Reading vocab")
##    vocabulary.read()
#    
    import model
#    try:
#        print >> sys.stderr, ("Trying to read training state for %s %s..." % (newkeystr, rundir))
#        (m, cnt, epoch, get_train_minibatch) = state.load(rundir, newkeystr)
#        print >> sys.stderr, ("...success reading training state for %s %s" % (newkeystr, rundir))
#        print >> sys.stderr, logfile
#        logging.basicConfig(filename=logfile, level=logging.DEBUG)
##        logging.basicConfig(filename=logfile, filemode="w", level=logging.DEBUG)
#        logging.info("CONTINUING FROM TRAINING STATE")
#    except IOError:
#        print >> sys.stderr, ("...FAILURE reading training state for %s %s" % (newkeystr, rundir))
#        print >> sys.stderr, ("INITIALIZING")
#
#        m = model.Model()
#        cnt = 0
#        epoch = 1
#        get_train_minibatch = examples.TrainingMinibatchStream()
#        logging.basicConfig(filename=logfile, filemode="w", level=logging.DEBUG)
#        logging.info("INITIALIZING TRAINING STATE")


    # TODO: Try to load old training state

    translation_model = {}
    for l1, l2 in HYPERPARAMETERS["W2W BICORPORA"]:
        translation_model[l1] = model.Model()

    # TODO: If we want more than one model, we should SHARE the embeddings parameters
    assert len(translation_model) == 1
    for l1 in HYPERPARAMETERS["W2W MONOCORPORA"]:
        assert 0

    cnt = 0
    epoch = 1
#    get_train_minibatch = examples.TrainingMinibatchStream()
    get_train_minibatch = w2w.examples.get_training_minibatch()
    logging.basicConfig(filename=logfile, filemode="w", level=logging.DEBUG)
    logging.info("INITIALIZING TRAINING STATE")


    logging.info(myyaml.dump(common.dump.vars_seq([hyperparameters, miscglobals])))

#    #validate(0)
#    diagnostics.diagnostics(cnt, m)
##    diagnostics.visualizedebug(cnt, m, rundir)
    while 1:
        logging.info("STARTING EPOCH #%d" % epoch)
        for ebatch in get_train_minibatch:
            cnt += len(ebatch)
#        #    print [wordmap.str(id) for id in e]

            for e in ebatch:
                # Make sure all examples have the same language
                assert e.l1 == ebatch[0].l1

            # The following is code for training on bilingual examples.
            # TODO: Monolingual examples?

            correct_sequences = []
            noise_sequences = []
            weights = []
            for e in ebatch:
                notw2, weight = e.corrupt
                correct_sequences.append(e.l1seq + [e.w2])
                noise_sequences.append(e.l1seq + [notw2])
                weights.append(weight)
            assert len(ebatch) == len(correct_sequences)
            assert len(ebatch) == len(noise_sequences)
            assert len(ebatch) == len(weights)

#            m.train(ebatch, noise_sequences, weights)
#            m.train(ebatch)
#
#            #validate(cnt)
#            if cnt % (int(1000./HYPERPARAMETERS["MINIBATCH SIZE"])*HYPERPARAMETERS["MINIBATCH SIZE"]) == 0:
#                logging.info("Finished training step %d (epoch %d)" % (cnt, epoch))
##                print ("Finished training step %d (epoch %d)" % (cnt, epoch))
#            if cnt % (int(100000./HYPERPARAMETERS["MINIBATCH SIZE"])*HYPERPARAMETERS["MINIBATCH SIZE"]) == 0:
#                diagnostics.diagnostics(cnt, m)
#                if os.path.exists(os.path.join(rundir, "BAD")):
#                    logging.info("Detected file: %s\nSTOPPING" % os.path.join(rundir, "BAD"))
#                    sys.stderr.write("Detected file: %s\nSTOPPING\n" % os.path.join(rundir, "BAD"))
#                    sys.exit(0)
#            if cnt % (int(HYPERPARAMETERS["VALIDATE_EVERY"]*1./HYPERPARAMETERS["MINIBATCH SIZE"])*HYPERPARAMETERS["MINIBATCH SIZE"]) == 0:
#                state.save(m, cnt, epoch, get_train_minibatch, rundir, newkeystr)
#                diagnostics.visualizedebug(cnt, m, rundir, newkeystr)
##                validate(cnt)
#        get_train_minibatch = examples.TrainingMinibatchStream()
        get_train_minibatch = w2w.examples.get_training_minibatch()
        epoch += 1

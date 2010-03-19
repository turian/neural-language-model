#!/usr/bin/env python

import sys
import string
import common.dump
from common.file import myopen
from common.stats import stats
from common.str import percent

import miscglobals
import logging

import w2w.examples
import diagnostics
import state

import cPickle

def validate(translation_model, cnt):
    import math
#    logranks = []
#    logging.info("BEGINNING VALIDATION AT TRAINING STEP %d" % cnt)
#    logging.info(stats())
    i = 0
    tot = 0
    correct = 0
    for (i, ve) in enumerate(w2w.examples.get_all_validation_examples_cached()):
        correct_sequences, noise_sequences, weights = ebatch_to_sequences([ve])
        source_language = ve.l1
        is_correct = translation_model[source_language].validate_errors(correct_sequences, noise_sequences)
#        print r
        for w in weights: assert w == 1.0

        tot += 1
        if is_correct: correct += 1

        if i % 1000 == 0: logging.info("\tvalidating %d examples done..." % i)
#    logging.info("Validation of model %s at cnt %d: validation err %s" % (translation_model[source_language].modelname, cnt, percent(correct, tot)))
    logging.info("VALIDATION of model at cnt %d: validation accuracy %s" % (cnt, percent(correct, tot)))
##        logging.info([wordmap.str(id) for id in ve])
#        logranks.append(math.log(m.validate(ve)))
#        if (i+1) % 10 == 0:
#            logging.info("Training step %d, validating example %d, mean(logrank) = %.2f, stddev(logrank) = %.2f" % (cnt, i+1, numpy.mean(numpy.array(logranks)), numpy.std(numpy.array(logranks))))
#            logging.info(stats())
#    logging.info("FINAL VALIDATION AT TRAINING STEP %d: mean(logrank) = %.2f, stddev(logrank) = %.2f, cnt = %d" % (cnt, numpy.mean(numpy.array(logranks)), numpy.std(numpy.array(logranks)), i+1))
#    logging.info(stats())
##    print "FINAL VALIDATION AT TRAINING STEP %d: mean(logrank) = %.2f, stddev(logrank) = %.2f, cnt = %d" % (cnt, numpy.mean(numpy.array(logranks)), numpy.std(numpy.array(logranks)), i+1)
##    print stats()

def ebatch_to_sequences(ebatch):
    """
    Convert example batch to sequences.
    """
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
    return correct_sequences, noise_sequences, weights

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

    # Random wait if we are a batch job
    import time
    if not HYPERPARAMETERS["console"]:
        wait = 100 * random.random()
        print >> sys.stderr, "Waiting %f seconds..." % wait
        time.sleep(wait)

#    import vocabulary
##    logging.info("Reading vocab")
##    vocabulary.read()
#    
    import model
    try:
        print >> sys.stderr, ("Trying to read training state for %s %s..." % (newkeystr, rundir))
        (translation_model, cnt, lastcnt, epoch) = state.load(rundir, newkeystr)
        print >> sys.stderr, ("...success reading training state for %s %s" % (newkeystr, rundir))
        print >> sys.stderr, logfile
        print >> sys.stderr, "CONTINUING FROM TRAINING STATE"
    except IOError:
        print >> sys.stderr, ("...FAILURE reading training state for %s %s" % (newkeystr, rundir))
        print >> sys.stderr, ("INITIALIZING")

        translation_model = {}
        print >> sys.stderr, "Loading initial embeddings from %s" % HYPERPARAMETERS["INITIAL_EMBEDDINGS"]
        # TODO: If we want more than one model, we should SHARE the embeddings parameters
        embeddings = cPickle.load(common.file.myopen(HYPERPARAMETERS["INITIAL_EMBEDDINGS"]))

        print >> sys.stderr, "INITIALIZING TRAINING STATE"

        all_l1 = {}
        for l1, l2 in HYPERPARAMETERS["W2W BICORPORA"]: all_l1[l1] = True
        for l1 in all_l1:
            translation_model[l1] = model.Model(modelname="translate-from-%s" % l1, window_size=HYPERPARAMETERS["WINDOW_SIZE"]+1, initial_embeddings=embeddings)
        # TODO: I'd like to free this memory, but translation_model doesn't make a copy.
#        embeddings = None
        cnt = 0
        lastcnt = 0
        epoch = 1
#        get_train_minibatch = examples.TrainingMinibatchStream()

    if HYPERPARAMETERS["console"]:
        print >> sys.stderr, "Console mode (not batch mode)."
        logging.basicConfig(level=logging.INFO)
    else:
        print >> sys.stderr, "YOU ARE RUNNING IN BATCH, NOT CONSOLE MODE. THIS WILL BE THE LAST MESSAGE TO STDERR."
        logging.basicConfig(filename=logfile, filemode="w", level=logging.INFO)

    assert len(translation_model) == 1
    for l1 in HYPERPARAMETERS["W2W MONOCORPORA"]:
        assert 0

#    get_train_minibatch = w2w.examples.get_training_minibatch_online()
    get_train_minibatch = w2w.examples.get_training_minibatch_cached()

    logging.info(myyaml.dump(common.dump.vars_seq([hyperparameters, miscglobals])))

    validate(translation_model, 0)
#    diagnostics.diagnostics(cnt, m)
##    diagnostics.visualizedebug(cnt, m, rundir)
#    state.save(translation_model, cnt, lastcnt, epoch, rundir, newkeystr)
    while 1:
        logging.info("STARTING EPOCH #%d" % epoch)
        for ebatch in get_train_minibatch:
            lastcnt = cnt
            cnt += len(ebatch)
#        #    print [wordmap.str(id) for id in e]

            source_language = ebatch[0].l1
            for e in ebatch:
                # Make sure all examples have the same source language
                assert e.l1 == source_language

            # The following is code for training on bilingual examples.
            # TODO: Monolingual examples?

            correct_sequences, noise_sequences, weights = ebatch_to_sequences(ebatch)
            translation_model[source_language].train(correct_sequences, noise_sequences, weights)

            #validate(translation_model, cnt)
            if int(cnt/1000) > int(lastcnt/1000):
                logging.info("Finished training step %d (epoch %d)" % (cnt, epoch))
#                print ("Finished training step %d (epoch %d)" % (cnt, epoch))
            if int(cnt/10000) > int(lastcnt/10000):
                for l1 in translation_model:
                    diagnostics.diagnostics(cnt, translation_model[l1])
                if os.path.exists(os.path.join(rundir, "BAD")):
                    logging.info("Detected file: %s\nSTOPPING" % os.path.join(rundir, "BAD"))
                    sys.stderr.write("Detected file: %s\nSTOPPING\n" % os.path.join(rundir, "BAD"))
                    sys.exit(0)
            if int(cnt/HYPERPARAMETERS["VALIDATE_EVERY"]) > int(lastcnt/HYPERPARAMETERS["VALIDATE_EVERY"]):
                validate(translation_model, cnt)
                pass
#                for l1 in translation_model:
#                    diagnostics.visualizedebug(cnt, translation_model[l1], rundir, newkeystr)

        validate(translation_model, cnt)
#        get_train_minibatch = w2w.examples.get_training_minibatch_online()
        get_train_minibatch = w2w.examples.get_training_minibatch_cached()
        epoch += 1

        state.save(translation_model, cnt, lastcnt, epoch, rundir, newkeystr)
#       validate(cnt)

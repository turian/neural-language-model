"""
Verbose debug output for the model.
"""

import logging
from common.stats import stats
from common.str import percent

import examples

import numpy
import random

def diagnostics(cnt, model):
    logging.info(stats())
    idxs = range(model.parameters.vocab_size)
    random.shuffle(idxs)
    idxs = idxs[:100]

    embeddings_debug(model.parameters.embeddings[idxs], cnt, "rand 100 words, model %s" % model.name)
    embeddings_debug(model.parameters.embeddings[:100], cnt, "top  100 words, model %s" % model.name)
    embeddings_debug(model.parameters.embeddings[model.parameters.vocab_size/2-50:model.parameters.vocab_size/2+50], cnt, "mid  100 words, model %s" % model.name)
    embeddings_debug(model.parameters.embeddings[-100:], cnt, "last 100 words, model %s" % model.name)
    weights_debug(model.parameters.hidden_weights.value, cnt, "hidden weights, model %s" % model.name)
    weights_debug(model.parameters.output_weights.value, cnt, "output weights, model %s" % model.name)
    logging.info(stats())

def visualizedebug(cnt, model, rundir, newkeystr, WORDCNT=500):
    idxs = range(model.parameters.vocab_size)
    random.shuffle(idxs)
    idxs = idxs[:WORDCNT]

    visualize(cnt, model, rundir, idxs, "randomized%s" % newkeystr)
    visualize(cnt, model, rundir, range(WORDCNT), "mostcommon%s" % newkeystr)
    visualize(cnt, model, rundir, range(-1, -WORDCNT*50, -1*50), "leastcommon%s" % newkeystr)
    visualize(cnt, model, rundir, range(model.parameters.vocab_size/2-WORDCNT*20/2,model.parameters.vocab_size/2+WORDCNT*20/2, 20), "midcommon%s" % newkeystr)

def visualize(cnt, model, rundir, idxs, str):
    """
    Visualize a set of examples using t-SNE.
    """
    from vocabulary import wordmap
    PERPLEXITY=30

    idxs = [id % model.parameters.embeddings.shape[0] for id in idxs]
    x = model.parameters.embeddings[idxs]
    print x.shape
    titles = [`wordmap().str(id)` for id in idxs]

    import os.path
    filename = os.path.join(rundir, "embeddings.model-%s.-%s-%d.png" % (model.name, str, cnt))
    try:
        from textSNE.calc_tsne import tsne
#       from textSNE.tsne import tsne
        out = tsne(x, perplexity=PERPLEXITY)
        from textSNE.render import render
        render([(title, point[0], point[1]) for title, point in zip(titles, out)], filename)
    except IOError:
        logging.info("ERROR visualizing", filename, ". Continuing...")

def embeddings_debug(w, cnt, str):
    """
    Output the l2norm mean and max of the embeddings, including in debug out the str and training cnt
    """
    totalcnt = numpy.sum(numpy.abs(w) >= 0)
    notsmallcnt = numpy.sum(numpy.abs(w) >= 0.1)
    logging.info("%d %s dimensions of %s have absolute value >= 0.1" % (cnt, percent(notsmallcnt, totalcnt), str))
    notsmallcnt = numpy.sum(numpy.abs(w) >= 0.01)
    logging.info("%d %s dimensions of %s have absolute value >= 0.01" % (cnt, percent(notsmallcnt, totalcnt), str))

    l2norm = numpy.sqrt(numpy.square(w).sum(axis=1))
    median = numpy.median(l2norm)
    mean = numpy.mean(l2norm)
    std = numpy.std(l2norm)
#    print("%d l2norm of top 100 words: mean = %f stddev=%f" % (cnt, numpy.mean(l2norm), numpy.std(l2norm),))
    l2norm = l2norm.tolist()
    l2norm.sort()
    l2norm.reverse()
    logging.info("%d l2norm of %s: median = %f mean = %f stddev=%f top3=%s" % (cnt, str, median, mean, std, `l2norm[:3]`))
#    print("top 5 = %s" % `l2norm[:5]`)

def weights_debug(w, cnt, str):
    """
    Output the abs median, mean, and max of the weights w, including in debug out the str and training cnt
    """
    w = numpy.abs(w)
    logging.info("%d abs of %s: median=%f mean=%f stddev=%f" % (cnt, str, numpy.median(w), numpy.mean(w), numpy.std(w),))
#    print("%d l2norm of top 100 words: mean = %f stddev=%f" % (cnt, numpy.mean(l2norm), numpy.std(l2norm),))
#    w = w.tolist()
#    w.sort()
#    w.reverse()
#    logging.info("\ttop 5 = %s" % `w[:5]`)
#    print("top 5 = %s" % `l2norm[:5]`)

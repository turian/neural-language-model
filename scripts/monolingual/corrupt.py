"""
Methods for corrupting examples.
"""

def corrupt_example(model, e):
    """
    Return a corrupted version of example e, plus the weight of this example.
    """
    from hyperparameters import HYPERPARAMETERS
    import random
    import copy
    e = copy.copy(e)
    last = e[-1]
    cnt = 0
    while e[-1] == last:
        if HYPERPARAMETERS["NGRAM_FOR_TRAINING_NOISE"] == 0:
            e[-1] = random.randint(0, model.parameters.vocab_size-1)
            pr = 1./model.parameters.vocab_size
        elif HYPERPARAMETERS["NGRAM_FOR_TRAINING_NOISE"] == 1:
            import noise
            from common.myrandom import weighted_sample
            e[-1], pr = weighted_sample(noise.indexed_weights())
#            from vocabulary import wordmap
#            print wordmap.str(e[-1]), pr
        else:
            assert 0
        cnt += 1
        # Backoff to 0gram smoothing if we fail 10 times to get noise.
        if cnt > 10: e[-1] = random.randint(0, model.parameters.vocab_size-1)
    weight = 1./pr
    return e, weight

def corrupt_examples(model, correct_sequences):
    noise_sequences = []
    weights = []
    for e in correct_sequences:
        noise_sequence, weight = model.corrupt_example(e)
        noise_sequences.append(noise_sequence)
        weights.append(weight)
    return noise_sequences, weights

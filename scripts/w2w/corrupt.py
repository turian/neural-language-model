"""
Methods for corrupting examples.
"""

from w2w.targetvocabulary import targetmap
from w2w.vocabulary import language

def corrupt_bilingual_example((l1, seq), w2):
    """
    Return a corrupted version of example as ((l1, seq), notw2), plus the weight of this example.
    """
    from hyperparameters import HYPERPARAMETERS
    import random
    import copy
    notw2 = w2
    cnt = 0
    while w2 == notw2:
        if HYPERPARAMETERS["NGRAM_FOR_TRAINING_NOISE"] == 0:
            e[-1] = random.randint(0, targetmap()[
            pr = 1./model.parameters.vocab_size
        elif HYPERPARAMETERS["NGRAM_FOR_TRAINING_NOISE"] == 1:
            assert 0
#            import noise
#            from common.myrandom import weighted_sample
#            e[-1], pr = weighted_sample(noise.indexed_weights())
##            from vocabulary import wordmap
##            print wordmap.str(e[-1]), pr
        else:
            assert 0
        cnt += 1
        # Backoff to 0gram smoothing if we fail 10 times to get noise.
        if cnt > 10: w2 = random.randint(0, model.parameters.vocab_size-1)
    weight = 1./pr
    return e, weight

def corrupt_bilingual_examples(batch):
    """
    Corrupt a minibatch of bilingual examples.
    Return noise_sequences, weights
    """
    # Make sure every example in the batch has the same source language.
    for (l1, seq), w2 in batch:
        assert l1 == batch[0][0][0]

    noise_sequences = []
    weights = []
    for (l1, seq), w2 in batch:
        noise_sequence, weight = model.corrupt_bilingual_example((l1, seq), w2)
        noise_sequences.append(noise_sequence)
        weights.append(weight)
    return noise_sequences, weights

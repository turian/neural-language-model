Approach based upon language model in Bengio et al ICML 09 "Curriculum Learning".

WHY ARE TRAINING UPDATES OCCURRING EVEN WHEN LOSS IS 0?
WHY DO OUTPUT SCORES SEEM TO DIVERGE?

TODO:
    * sqrt scaling of SGD updates
    * Use normalization of embeddings?
    * How do we initialize embeddings?
    * Use tanh, not softsign?
    * When doing SGD on embeddings, use sqrt scaling of embedding size?

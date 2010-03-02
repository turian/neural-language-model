Approach based upon language model in Bengio et al ICML 09 "Curriculum Learning".

You will need my common python library:
    http://github.com/turian/common
and my textSNE wrapper for t-SNE:
    http://github.com:turian/textSNE

To train a monolingual language model, probably you should run:
    [edit hyperparameters.language-model.yaml]
    ./build-vocabulary.py
    ./train.py

To train word-to-word multilingual model, probably you should run:
    [TODO:
    * Initialize using monolingual language model in source language.
    * More than one target language.
    * Loss = logistic, not margin.
    ]

    # [optional: Run the following if your alignment for language pair l1-l2
    # is in form l2-l1]
    ./scripts/preprocess/reverse-alignment.pl

    ./w2w/build-vocabulary.py
    ./w2w/build-target-vocabulary.py
    ./train-w2w.py

TODO:
    * sqrt scaling of SGD updates
    * Use normalization of embeddings?
    * How do we initialize embeddings?
    * Use tanh, not softsign?
    * When doing SGD on embeddings, use sqrt scaling of embedding size?

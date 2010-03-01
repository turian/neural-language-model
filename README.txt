Approach based upon language model in Bengio et al ICML 09 "Curriculum Learning".

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
    ./w2w/build-vocabulary.py
    ./w2w/build-target-vocabulary.py
    ./train-w2w.py

TODO:
    * sqrt scaling of SGD updates
    * Use normalization of embeddings?
    * How do we initialize embeddings?
    * Use tanh, not softsign?
    * When doing SGD on embeddings, use sqrt scaling of embedding size?

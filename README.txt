Approach based upon language model in Bengio et al ICML 09 "Curriculum Learning".


You will need my common python library:
    http://github.com/turian/common
and my textSNE wrapper for t-SNE:
    http://github.com:turian/textSNE

You will need Murmur for hashing.
    easy_install Murmur

To train a monolingual language model, probably you should run:
    [edit hyperparameters.language-model.yaml]
    ./build-vocabulary.py
    ./train.py

To train word-to-word multilingual model, probably you should run:
    cd scripts; ln -s hyperparameters.language-model.sample.yaml s hyperparameters.language-model.yaml

    # Create validation data:
    ./preprocess-validation.pl > ~/data/SemEval-2-2010/Task\ 3\ -\ Cross-Lingual\ Word\ Sense\ Disambiguation/validation.txt Tokenizer v3

    # [optional: Lemmatize]
    Tadpole --skip=tmp -t ~/dev/python/mt-language-model/neural-language-model/data/filtered-full-bilingual/en-nl/filtered-training.nl | perl -ne 's/\t/ /g; print lc($_);' | chop 3 | from-one-line-per-word-to-one-line-per-sentence.py > ~/dev/python/mt-language-model/neural-language-model/data/filtered-full-bilingual-lemmas/en-nl/filtered-training-lemmas.nl
    #

    [TODO:
    * Initialize using monolingual language model in source language.
    * Loss = logistic, not margin.
    ]

    # [optional: Run the following if your alignment for language pair l1-l2
    # is in form l2-l1]
    ./scripts/preprocess/reverse-alignment.pl

    ./w2w/build-vocabulary.py
    # Then see the output with ./w2w/dump-vocabulary.py, to see if you want
    # to adjust the w2w minfreq hyperparameter

    ./w2w/build-target-vocabulary.py
    # Then see the output with ./w2w/dump-target-vocabulary.py

    ./w2w/build-initial-embeddings.py

    # [optional: Filter the corpora only to include sentences with certain
    # focus words.]
    # You want to make sure this happens AFTER
    # ./w2w/build-initial-embeddings.py, so you have good embeddings for words
    # that aren't as common in the filtered corpora.
    ./scripts/preprocess/filter-sentences-by-lemma.py
    # You should then move the filtered corpora to a new data directory.]

    #[optional: This will cache all the training examples onto disk. This will
    # happen automatically during training anyhow.]
    ./scripts/w2w/build-example-cache.py

    ./w2w/train.py

TODO:
    * sqrt scaling of SGD updates
    * Use normalization of embeddings?
    * How do we initialize embeddings?
    * Use tanh, not softsign?
    * When doing SGD on embeddings, use sqrt scaling of embedding size?

"""
Lemmatize English using the NLTK WordNetLemmatizer.
"""

from nltk.stem.wordnet import WordNetLemmatizer

_lmtzr = None
def lmtzr():
    global _lmtzr
    if _lmtzr is None: _lmtzr = WordNetLemmatizer()
    return _lmtzr

def lemmatize(language, wordform):
    assert language == "en"
    return lmtzr().lemmatize(wordform)

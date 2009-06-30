allwords.gz is from Childes corpus, Eng-USA/allwords.gz

Create vocabulary:
zcat allwords.gz | sort | uniq -c | sort -rn > allwords.vocabulary.txt

test is the first 10K words.
validation is the next 10K words.
train is the rest.

=============

wikitext.txt.gz is preprocessed English wikipedia, broken into sentences and
tokenized and shuffled.

ls | grep gz | ~/common/scripts/shuffle.sh | xargs zcat | ../../scripts/preprocess.pl  | grep . | ~/common/scripts/shuffle.sh | gzip -c > ../wikitext.txt.gz

zcat wikitext.txt.gz | head -10000 | gzip -c > wikitext.test.txt.gz
zcat wikitext.txt.gz | head -20000 | tail -10000 | gzip -c > wikitext.validation.txt.gz
zcat wikitext.txt.gz | tail -66151742 | gzip -c > wikitext.train.txt.gz

=============

italian-wikitext.txt.gz is preprocessed Italian wikipedia:

bzcat ~/data/italian_SemaWiki_attardi/3_tokenized.txt.bz2 | ~/data/italian_SemaWiki_attardi/one-sentence-per-line.pl | ../scripts/preprocess.pl  | grep . | ~/common/scripts/shuffle.sh | gzip -c > italian-wikitext.txt.gz

zcat italian-wikitext.txt.gz | head -10000 | gzip -c > italian-wikitext.test.txt.gz
zcat italian-wikitext.txt.gz | head -20000 | tail -10000 | gzip -c > italian-wikitext.validation.txt.gz
zcat italian-wikitext.txt.gz | tail -5672365 | gzip -c > italian-wikitext.train.txt.gz

# Sanity check
zcat italian-wikitext.test.txt.gz italian-wikitext.validation.txt.gz italian-wikitext.train.txt.gz | md5sum
zcat italian-wikitext.txt.gz | md5sum

../scripts/examples.py italian-wikitext.validation.txt.gz | ~/common/scripts/shuffle.sh | head -1000 | gzip -c > italian-wikitext.validation-1000.txt.gz 


# Vocabulary
zcat italian-wikitext.train.txt.gz | perl -ne 's/ /\n/g; print' | grep . | sort | uniq -c | sort -rn | gzip -c > vocabulary-italian-wikitext.txt.gz
zcat vocabulary-italian-wikitext.txt.gz | head -20000 | gzip -c > vocabulary-italian-wikitext-20000.txt.gz

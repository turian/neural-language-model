allwords.gz is from Childes corpus, Eng-USA/allwords.gz

Create vocabulary:
zcat allwords.gz | sort | uniq -c | sort -rn > allwords.vocabulary.txt

test is the first 10K words.
validation is the next 10K words.
train is the rest.

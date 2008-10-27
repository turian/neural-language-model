allwords.gz is from Childes corpus, Eng-USA/allwords.gz

Create vocabulary:
zcat allwords.gz | sort | uniq -c | sort -rn > allwords.vocabulary.txt


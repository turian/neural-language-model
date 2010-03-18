#!/usr/bin/perl -w
#
#  Transform the validation data into a form that it can be used by the system.
#

$VDIR = "/u/turian/data/SemEval-2-2010/Task 3 - Cross-Lingual Word Sense Disambiguation";

foreach $f (`find '$VDIR' -name \*.data`) {
    open(F, "<$f") or die $!;
    while (<F>) {
        $lemma = $1 if /<lexelt item="(.*)\.n">/;
        if (/<context>(.*)<\/context>/) {
            $l = $1;
            $l =~ s/<head>[^<>]*<\/head>/$lemma/g;
            open(O, "| ~/data/europarl-v5/europarl/tools/tokenizer.perl -l en | ~/data/europarl-v5/preprocessed/lowercase.perl | ~/utils/src/treetagger-3.2/l.py en > /tmp/removeme.txt");
            print O $l;
            $l = `cat /tmp/removeme.txt`;
            print $l;
        }
    }
}

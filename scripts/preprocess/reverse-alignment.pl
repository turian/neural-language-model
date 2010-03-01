#!/usr/bin/perl -w
#
#  USAGE:
#       ./reverse-alignment.pl filename.align.l1-l2 [...]
#
#  Create a file filename.align.l2-l1 with the alignments reversed.
#

die $! unless scalar @ARGV >= 1;

foreach $f (@ARGV) {
    if ($f =~ m/(.*\.align\.)(..)-(..)$/) {
        $fnew = "$1$3-$2";
    } else {
        die $!;
    }
   
    if (-e $fnew) {
        print "$fnew already exists";
        next;
    }
    
    $cmd = "cat $f | perl -ne 's/(\\d+)-(\\d+)/\$2-\$1/g; print' > $fnew";
    print "$cmd\n";
    system("$cmd");

    print "SANITY CHECK... (shouldn't see any output after this command)\n";
    $cmd = "cat $fnew | perl -ne 's/(\\d+)-(\\d+)/\$2-\$1/g; print' | diff - $f";
    print "$cmd\n";
    system("$cmd");
}

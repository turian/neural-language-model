#!/usr/bin/perl -w
#
# For each directory in @ARGV, go in that directory and remove every
# model file except for the last one.
#

@torm = ();
foreach $d (@ARGV) {
    $last = -1;
    # Find the last model
    foreach $f  (split(/[\r\n]+/, `ls $d`)) {
        if ($f =~ m/model-(\d+).pkl/) {
            $last = $1 if $1 > $last;
        }
    }
    # All non-last models are added to torm
    foreach $f  (split(/[\r\n]+/, `ls $d`)) {
        if ($f =~ m/model-(\d+).pkl/) {
            if ($1 < $last) {
                $torm[++$#torm] = "$d/$f";
            } else {
                print "KEEPING $d/$f\n";
            }
        }
    }
}
foreach $f (@torm) {
    $cmd = "rm $f";
    print "$cmd\n";
    system("$cmd");
}

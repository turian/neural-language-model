#!/usr/bin/perl -w
#
#  Make a .dat file for each .out file.
#

$gnuplot = "plot";
$first = 1;
#foreach $f (split(/[\r\n]+/, `ls [0-9]*out`)) {
foreach $f (split(/[\r\n]+/, `ls ../run*/log.* | grep -v 'dat\$'`)) {
    next if not $f =~ m/f1426d05c578bfd029875b646b66195044/;
    next if $f =~ m/\.dat$/;
    ($badf = $f) =~ s/\/[^\/]*$/\/BAD/;
    next if -e $badf;
    ($fnew = $f) =~ s/$/-trainerror.dat/;
    die $! if $fnew eq $f;
    print STDERR "$f => $fnew\n";
    # We can allow e to be grepped, because of numbers like 5e-8
    $cmd = "cat $f | grep --text 'pre-update train err' | perl -ne 's/=/ /g; print' | cut -d ' ' -f 2,10 | grep -v '[a-df-zA-DF-Z]' | grep '0000 ' > $fnew";
    print STDERR "$cmd\n";
    system($cmd);
    $gnuplot .= "," unless $first;
    $first = 0;
    $gnuplot .= " \\\n\t'$fnew' with lp"
}
print "$gnuplot\n";

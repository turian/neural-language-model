#!/bin/sh

rm *trainerror.dat
#rm ../run*/*trainerror.dat

# Make all dat files
../../eda/make-graphs-trainerror.pl

ln -s ../*/*trainerror.dat .

# Sort all dat files
# First perl recipe adds gnuplot codes
# Second perl recipe strips final ', \'  to prevent gnuplot error
echo > graphs-trainerror.gp
echo "set terminal postscript color 12" >> graphs-trainerror.gp
echo "set output 'graphs-trainerror.ps'" >> graphs-trainerror.gp
echo "set logscale y" >> graphs-trainerror.gp
echo "plot [] [] \\" >> graphs-trainerror.gp
~/dev/common-scripts/sort-curves.py *trainerror.dat | perl -ne "chop; print \"\\t'\$_' with l lw 3, \\\\\\n\"" | perl -e '$str = ""; while(<>){ $str .= $_; } $str =~ s/, \\$//s; print $str' >> graphs-trainerror.gp

gnuplot graphs-trainerror.gp
ps2pdf graphs-trainerror.ps
cp *pdf ~/public_html/priv ; chmod a+r ~/public_html/priv/*pdf
#scp  *pdf turian@joyeux.iro.umontreal.ca:public_html/priv/

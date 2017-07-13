#!/usr/bin/env bash

F=FNS
T=basic.plot
OUT=OUT.eps

I=#

cat > $T <<EOF

${I}set term post eps size 3.5, 2.5 
${I}set output '$OUT'

set key left top

set xlabel 'Nodes Added by the Playout'
set ylabel 'Normalized Number of Samples'

set grid
#set log y
set style data linespoints
#set xtics (1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
#set mytics 
#set format y "%e"
set ytics 0,0.1,1

plot \\
EOF


FROM=$1
TO=$(($FROM + 9))

for fn in $( sed -n "${FROM},${TO}p" $F ) ; do
echo $fn
#echo "'$fn' using 1:2 with lines, \\" >> $T
echo "'$fn' smooth cumulative, \\" >> $T
done

sed -i '$s/,.*$//' $T


[ "$I" = '#' ] && { cat $T - | gnuplot ; } || gnuplot $T

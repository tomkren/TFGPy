#!/bin/sh

PROC=4
REPEAT=10
K=5

for LEVEL in 2 3 ; do
    for DOMAIN in 'stack' 'app-tree' ; do

        date  >> LOG_run.sh
        echo $LEVEL $DOMAIN  >> LOG_run.sh
        ./run_experiment.py "--k=$K" "--proc=$PROC" "--repeat=$REPEAT" "--${DOMAIN}" --nmcs "--nmcs-level=$LEVEL"  | tee "LOG_${DOMAIN}_nmcs_level_$LEVEL"
    done
done
date  >> LOG_run.sh

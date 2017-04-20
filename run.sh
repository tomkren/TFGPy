#!/bin/sh

python3 test_norm.py > LOG

rm .LOG_nop.swp
rm .LOG_nf.swp

cat LOG | sed -n '/NOP/,/NOPEND/p' > LOG_nop
cat LOG | sed -n '/NF/,/NFEND/p' > LOG_nf

#gvim -O LOG_nop LOG_nf

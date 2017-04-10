#!/bin/sh

for a in test*.py  ; do
	echo $a
	python3 $a
done

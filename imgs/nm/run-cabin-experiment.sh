#!/bin/bash

# This file is suitable for running i2c together with neuralmonkey on a Windows machine.
# We use the Windows 10 build in Ubuntu to do this. 
# For simple workflow we have prepared a windows directory connecting Win with Ubuntu by:
# ln -s /mnt/c/Users/sekol/Documents/ubuntu-cabin/ cabin
# .. which created a symlink ~/cabin/ 

# (1) Run i2c.py and let it fill the ubuntu-cabin with generated experiment
# (2) Run this script to execute the prepared experiment.

# This scipt can be run from any place as long as the following tree structure is respected:

# Expected dir tree:
# ~/cabin/experiment ... prepared experiment data end nm-configs from i2c script.
# ~/cabin/results    ... dir for storing results (but wil be created, so no worries about this little bird)
# ~/experiments      ... dir for storing results inside ubuntu (more persistent backup) (but wil be created, so no worries about this little bird)
# ~/nm               ... neural-monkey venv
# ~/neuralmonkey     ... neural-monkey repo

# ALSO: Please pay attention to the following lame fact: experiment.ini currently contains full path to experiment dir, 
# i.e. prefix="/home/sekol/experiment" 


cd ~


if [ -d experiment/ ]; then

	if [ ! -d experiments/ ]; then
		echo Crating new dir for old experiment results in the home.
		mkdir experiments/
	fi

	echo Moving experiment dir to: experiments/$(date +%s)/
	mv experiment/ experiments/$(date +%s)/
else
	echo No dir experiment found in ~.
fi

echo Copying experiment from the Cabin.
cp -R cabin/experiment/ experiment/

echo Activating neural-monkey venv.
source nm/bin/activate

# Change to  ~/experiment
cd experiment/

echo starting The Experiment ...
../neuralmonkey/bin/neuralmonkey-train experiment.ini


deactivate


# back to the ~
cd ..

if [ ! -d cabin/results/ ]; then
	echo Crating new dir for experiment results in the Cabin.
	mkdir cabin/results/
fi

echo Copying updated experiment dir with results to the Cabin: cabin/results/$(date +%s)/
cp -R experiment/ cabin/results/$(date +%s)/


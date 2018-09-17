#!/bin/bash

# It is not a good idea to run this scipt directly, since it is probable that it will fail for some reason,
# and it won't react to this fail. Instead, it is recommanded to manually execute each non-echo line.

echo Until this script is properly tested, we exit right after the start to prevent any damage
exit

echo Go to home dir ...
cd ~

echo this is probably already done, but for the record ...
git clone https://github.com/tomkren/TFGPy.git


echo Python version is ...
python3 -V

echo Creating venv for neural monkey ...
python3 -m venv nm

echo Switch to the newly created nm venv ...
source nm/bin/activate

echo TODO: make TFGPy requrements.txt and add the following packages to it!
echo Install Pillow, ImageHash and matplotlib: python packages used in TFGPy in i2c ...
pip install Pillow
pip install ImageHash
pip install matplotlib

echo Omg, for making matplotlib work it was also needed to install python3-tk on my win ubuntu.. 
sudo apt-get install python3-tk 


echo Clone neural monkey repo ...
git clone https://github.com/ufal/neuralmonkey

echo Install distribute thingy, a problematic prerequisity not in requirements.txt ...
easy_install distribute

echo Install numpy separetly, for some reason this is neededed oterwise requirements.txt install fails ...
pip install numpy

echo Go to neural monkey dir ...
cd neuralmonkey/

echo Install the rest of neural monkey prerequisities ...
pip install --upgrade -r requirements.txt

echo Back to the home dir ...
cd ..





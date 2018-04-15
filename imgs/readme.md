# i2c imgs #

This directory is a place where the generated images are saved.
Currently, the script generating them is the `draw.py` script.
These images are intended for neural network experiments.


## Notes ##

- idea : concat more hashes ...






## Sampling generating possibilities ##

- simple_sampling: sample fixed number of times, without consideration of number of fails 
- try to generate fix number of trees, with som limits that may fail due to too much attempts




## Experiment: Influence of options on num gen trees


### 32, hash_size=8 

Num Generated Images: 242
Generating Time: 6.80 s
Stats fo Sizes:
Tree size Num of all trees    New trees
1         2                   2
3         8                   4
5         80                  26
7         1024                210

Num Generated Images: 22880
Generating Time: 1666.82 s
Size Stats:
size      num                 new
1         2                   2
3         8                   4
5         80                  26
7         1024                210
9         14848               2054
11        231936              20584



### 32, hash_size=16 

Num Generated Images: 264
Generating Time: 7.21 s
Image size: 32×32
Stats fo Sizes:
Tree size      Num of all trees    New trees
1              2                   2
3              8                   4
5              80                  26
7              1024                232

Num Generated Images: 2602
Generating Time: 136.94 s (mozna nepresne, bezelo toho vic zrovna)
Image size: 32×32
pHash size: 16
Stats for Sizes:
Tree size      Num of all trees    New trees
1              2                   2
3              8                   4
5              80                  26
7              1024                232
9              14848               2338

### 32, hash_size=32, highfreq_factor=4

Num Generated Images: 2612
Generating Time: 136.40 s
Image size: 32×32
pHash size: 32
Stats for Sizes:
Tree size      Num of all trees    New trees
1              2                   2
3              8                   4
5              80                  26
7              1024                232
9              14848               2348

Num Generated Images: 28148
Generating Time: 1621.01 s
Image size: 32×32
pHash size & highfreq_factor: 32, 4
Stats for Sizes:
Tree size      Num of all trees    New trees           New/All %
1              2                   2                   100.00
3              8                   4                   50.00
5              80                  26                  32.50
7              1024                232                 22.66
9              14848               2348                15.81
11             231936              25536               11.01


### 32, 32, 8

Num Generated Images: 2608
Generating Time: 163.71 s
Image size: 32×32
pHash size & highfreq_factor: 32, 8
Stats for Sizes:
Tree size      Num of all trees    New trees
1              2                   2
3              8                   4
5              80                  26
7              1024                232
9              14848               2344


### 64, hash_size=8 

Num Generated Images: 258
Generating Time: 7.25 s
Image size: 64×64
Stats fo Sizes:
Tree size      Num of all trees    New trees
1              2                   2
3              8                   4
5              80                  26
7              1024                226

### 512, hash_size=16

Num Generated Images: 264
Generating Time: 11.57 s
Image size: 512×512
pHash size: 16
Stats for Sizes:
Tree size      Num of all trees    New trees
1              2                   2
3              8                   4
5              80                  26
7              1024                232

### 1024, hash_size=8 

Num Generated Images: 258
Generating Time: 27.85 s
Image size: 1024×1024
Stats fo Sizes:
Tree size      Num of all trees    New trees
1              2                   2
3              8                   4
5              80                  26
7              1024                226
# i2c imgs #

This directory is a place where the generated images are saved.
Currently, the script generating them is the `draw.py` script.
These images are intended for neural network experiments.


## experiment directory structure ##

`/`                       ... "path"; the experiment root folder ... e.g. `imgs/gen/`
`/imgs/`                  ... All generated imgs will be here.
`/imgs/%08d.png`          ... Single image filename pattern.
`imgs/handmade_examples/` ... Hand made examples, handy for visual chech that generator is ok.
`/imgs.txt`               ... Image filenames, one image per line.
`/train_imgs.txt`         ... Training sub-dataset: Image filenames, one image per line.
`/dev_imgs.txt`           ... Validation sub-dataset: Image filenames, one image per line.
`/prefix.txt`             ... Target codes of images in prefix notation, one image per line.
`/train_prefix.txt`       ... Training sub-dataset: Target codes of images in prefix notation, one image per line.
`/dev_prefix.txt`         ... Validation sub-dataset: Target codes of images in prefix notation, one image per line.
`/jsons.txt`              ... Target codes of images in json notation, one image per line.
`/stats.md`               ... Human readable stats in markdown, generated during the dataset generation process. 
`/roots.txt`              ... (probably deprecated) Just the root symbol (first prefix) of the code, one image per line.



## Notes ##

- idea : concat more hashes ...

- todo: enhance stats so that it appends during the run (header, body, footer approach)
- todo: do stats dát tabulku s:
        num_all_trees
        num_samples
        num_unique_trees
        num_new_phash_images
        asi % = num_new_phash_images / num_all_trees
   

max_tree_size -> num trees cumulative
 7  -> `~ 264`
 9  -> `~ 2602`
 11 -> `~ 28148`


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






---------------------------------








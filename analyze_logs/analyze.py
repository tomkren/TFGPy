#!/usr/bin/env python3

import numpy as np
import sys
from collections import OrderedDict
from collections import namedtuple

import random

#
# parse input
#

Info = namedtuple('Info', ['step', 'fitness', 'end_flag'])
log_flushed = OrderedDict()
log = {}

input = sys.stdin

SIZES = []

for line in input:
    tokens = line.strip().split('\t')
    assert len(tokens) in (3, 4)
    if len(tokens) == 4:
        assert tokens[3] == 'AT END'

    i = Info(int(tokens[1]), float(tokens[2]), len(tokens) == 4)
    c = tokens[0]

    if c in log_flushed:
        fl = log_flushed[c]
        print("WARN: %s DUPLICATE CACHE ID; #elements %d" % (c, len(fl)), file=sys.stderr)
        #print(fl[-2:])
        #print(i)
        c2 = "%s_dedup_stuff_%f" % (c, random.random())
        assert c2 not in log_flushed
        del log_flushed[c]
        log_flushed[c2] = fl

    cache = log.setdefault(c, [])
    if cache:
        last = cache[-1]
        assert not last.end_flag
        try:
            assert i.step > last.step or (i.end_flag and i.fitness == last.fitness)
            assert i.fitness >= last.fitness
            if i.fitness == last.fitness:
                pass
                assert i.end_flag
        except AssertionError:
            # print('skip,', end='', file=sys.stderr)
            continue

    cache.append(i)
    if i.end_flag:
        del log[c]
        assert c not in log_flushed
        SIZES.append(cache[-1].step)
        log_flushed[c] = cache

print(file=sys.stderr)
#
# fill in max_fitness for points without change
#

for c, cache in log.items():
    if cache:
        assert c not in log_flushed
        SIZES.append(cache[-1].step)
        log_flushed[c] = cache

##
#       kokotsky rozbity algoritmus
#

# print(len(log_flushed))

sz = np.array(SIZES)

bins = {}

NUM_BINS = 100
MIN, MAX = 0, sz.max()
ONE_BIN = MAX / NUM_BINS


def gb(i):
    return int(i.step // ONE_BIN)


for c, cache in log_flushed.items():
    assert cache
    if len(cache) == 1:
        print("WARN: skipping 1 sizec cache", file=sys.stderr)
        continue

    if len(cache) >= 2:
        ll, l = cache[-2:]
        if ll.fitness == l.fitness and ll.step == l.step:
            cache.pop()

    if False:
        print("=========", c, "=========", sep='\n')
        for i in cache:
            print(i.step, i.fitness)
            # continue

    last = None
    for i in cache:
        if last != None:
            for bin in range(gb(last) + 1, gb(i)):
                bins.setdefault(bin, []).append(last.fitness)

        bin = gb(i)
        bins.setdefault(bin, []).append(i.fitness)
        last = i

    if True:
        b = gb(last)
        for bin in range(b + 1, NUM_BINS):
            bins.setdefault(bin, []).append(last.fitness)

#
# avg
#

last = 0

for bin, l in sorted(bins.items()):
    #print()

    a = np.array(l)
    assert len(a)
    #print("jednicky", sum(a == 1.0))

    if bin >= 5:
        #break
        pass
    # print(a)
    if len(sys.argv) >= 2 and sys.argv[1] == 'ones':
        m = sum(a == 1.0) / len(l)
        sd = 0
    else:
        m = a.mean()
        sd = np.sqrt(((a - m) ** 2).mean())

    if bin > 0:
        print(bin,
              m ,
              #max(m, last),
              len(l))
              #sd)
    last = max(m, last)

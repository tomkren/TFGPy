#!/usr/bin/env pypy3
import random

def v1():
    random.seed(17)
    NSAMPLES = 5000000
    MAX_S = 50

    def one_sim(y, ps, limit):
        for N in range(1, limit):
            r = random.random()
            for steps, pst in ps:
                if r < pst:
                    y += steps
                    break
                else:
                    r -= pst

            assert y >= 0
            if y == 0:
                return N
        return limit - 1


    for ps, label in [
        ([(1, 0.5), (-1, 1)], "basic"),
        ([(1, 0.1), (-1, 1)], "a_lot_of_terminals"),
        #([(1, 4 / 39), (-1, 1)], "cazenave2013_primes"),
        #([(1, 1 / 5), (-1, 1)], "cazenave2013_algebra"),
        #([(1, 4 / 10), (-1, 1)], "cazenave2013_parity"),
        ([(1, 2 / 3), (-1, 1)], "cazenave2013_max"),
        #([(0, 5 / 16), (1, 9 / 16), (-1, 1)], "lim2016_add"),
        #([(0, 2 / 12), (1, 3 / 12), (2, 1 / 12), (-1, 1)], "lim2016_cmp"),
        #([(0, 2 / 16), (1, 3 / 16), (2, 1 / 16), (-1, 1)], "lim2016_grade"),
        #([(0, 2 / 10), (1, 3 / 10), (2, 1 / 10), (-1, 1)], "lim2016_median"),
    ]:
        starts = list(range(1,11)) # + list(range(11, MAX_S + 1, 5))
        for start_size in starts:
            sizes = {}
            for i in range(NSAMPLES):
                k = one_sim(start_size, ps, MAX_S + 1)
                sizes[k] = sizes.get(k, 0) + 1
            fn = 'sizes_s=%02d_%s.dat' % (start_size, label)
            with open(fn, 'w') as fout:
                first = None
                for i, c in sorted(sizes.items()):
                    if first is None:
                        # s.t. the plots do not start in the air
                        print(i-0.001, 0, file=fout)
                        first = i
                    print(i, c/NSAMPLES, file=fout)
            print(fn)


"""
def v1():
    random.seed(17)
    NSAMPLES = 1000000
    MAX_S = 40

    def one_sim(y, ps, limit):
        for N in range(1, limit):
            r = random.random()
            for steps, pst in ps:
                if r < pst:
                    y += steps
                    break
                else:
                    r -= pst

            assert y >= 0
            if y == 0:
                return N
        return limit - 1


    # TODO, pred prvni pridat nulu at ty ploty nezacijnaj ve vzduchu

    # TODO add regresion with a lot of terminals
    for ps, goal, label in [
        ([
# 0 int
# 1 float
# 2 boolean
# 3 string
# 4 void
# 5 control
# 6 cond
# 7 any

# functions
            (1, 3),
            (2, 1),

            # if
            ((-1, 2), (1, 1), (2, 2)),
            # strlen
            ((-1, 3), (1, 0)),
            # return bool: bool -> nop
            ((-1, 1), (1, 2)),
# terminals
            # load string
            ((-1, 0),),
            ((-1, 0),),
            ((-1, 0),),
            # true / false
            ((-1, 1),),
            ((-1, 1),),
            # nop
            ((-1, 2),)
                ], 7, "lim2016_cmp_2"),

   ]:
        starts = list(range(1,11)) #[1, 2, 3, 4, ] + list(range(5, MAX_S + 1, 5))
        for start_size in starts:
            sizes = {}
            for i in range(NSAMPLES):
                k = one_sim(start_size, ps, 51)
                sizes[k] = sizes.get(k, 0) + 1
            fn = 'sizes_s=%02d_%s.dat' % (start_size, label)
            with open(fn, 'w') as fout:
                for i, c in sorted(sizes.items()):
                    print(i, c/NSAMPLES, file=fout)
            print(fn)
"""


if __name__ == "__main__":
    v1()

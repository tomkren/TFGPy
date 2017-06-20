from collections import OrderedDict

import app_tree
import domain_koza_apptree
from fitness_cache import FitnessCache
from parsers import parse_typ, parse_ctx

size_d = {}


# https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n/3035188#3035188
def primes(n):
    """ Returns  a list of primes < n """
    sieve = [True] * n
    for i in range(3, int(n ** 0.5) + 1, 2):
        if sieve[i]:
            sieve[i * i::2 * i] = [False] * int(((n - i * i - 1) // (2 * i) + 1))
    return [2] + [i for i in range(3, n, 2) if sieve[i]]


def domain_primes():
    PRIME_LIMIT = 2000000
    PRIMES = set(primes(PRIME_LIMIT))

    global_symbols = {
        'plus': lambda x: (lambda y: x + y),
        'minus': lambda x: (lambda y: x - y),
        'times': lambda x: (lambda y: x * y),
        'rdiv': lambda p: (lambda q: p / q if q else 1),
    }

    R = 'R'
    numbers = [(str(i + 1), R) for i in range(10)]
    primes_lt_100 = primes(100)
    goal = parse_typ(R)

    stuff = [('plus', (R, '->', (R, '->', R))),
             ('minus', (R, '->', (R, '->', R))),
             ('times', (R, '->', (R, '->', R))),
             ('rdiv', (R, '->', (R, '->', R))),
             ('x', R), ('c', R)]# + numbers + [(str(p), R) for p in primes_lt_100]
    gamma = parse_ctx(OrderedDict(stuff))
    cache = FitnessCache()

    def fitness(individual_app_tree):
        global size_d
        size = individual_app_tree.count_nodes()[app_tree.Leaf]
        size_d[size] = size_d.get(size, 0) + 1

        s = "lambda x,c : %s" % individual_app_tree.eval_str()
        cres = cache.d.get(s, None)
        if cres is not None:
            return cres

        fun = eval(s, global_symbols)
        assert callable(fun)
        try:
            got = 0
            i = 0
            old = set()
            while True:
                candidate = fun(i, 0)
                assert candidate < PRIME_LIMIT
                if candidate not in PRIMES:
                    break
                if candidate not in old:
                    got += 1
                    old.add(candidate)
                i += 1

            score = got
        except (OverflowError, ValueError):
            score = 0.0

        cache.update(s, score)
        return score

    return goal, gamma, fitness, (lambda: len(cache)), cache


def make_env_app_tree():
    return domain_koza_apptree.make_env_app_tree(get_raw_domain=domain_primes,
                                                 early_end_limit=None)

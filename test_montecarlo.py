import math
from collections import OrderedDict

from parsers import parse_typ, parse_ctx


def regression_domain_koza_poly():
    def koza_poly(x):
        return x + x ** 2 + x ** 3 + x ** 4

    global_symbols = {
        'plus': lambda x: (lambda y: x + y),
        'minus': lambda x: (lambda y: x - y),
        'times': lambda x: (lambda y: x * y),
        'rdiv': lambda p: (lambda q: p / q if q else 1),
        'rlog': lambda x: math.log(abs(x)) if x else 0,
        'sin': math.sin,
        'cos': math.cos,
        'exp': math.exp,
    }

    R = 'R'
    goal = parse_typ(R)
    gamma = parse_ctx(OrderedDict([
        ('plus', (R, '->', (R, '->', R))),
        ('minus', (R, '->', (R, '->', R))),
        ('times', (R, '->', (R, '->', R))),
        ('rdiv', (R, '->', (R, '->', R))),
        ('sin', (R, '->', R)),
        ('cos', (R, '->', R)),
        ('exp', (R, '->', R)),
        ('rlog', (R, '->', R)),
        ('x', R),
    ]))

    cache_d = {}

    def fitness(individual_app_tree, target_f=koza_poly, num_samples=20):
        s = "lambda x : %s" % individual_app_tree.eval_str()
        if s in cache_d:
            return cache_d[s]

        fun = eval(s, global_symbols)
        assert callable(fun)
        samples = [-1 + 0.1 * i for i in range(num_samples)]
        try:
            error = sum(abs(fun(val) - target_f(val)) for val in samples)
        except OverflowError:
            return 0.0
        score = 1 / (1 + error)
        cache_d[s] = score

        return score

    return goal, gamma, fitness, (lambda: len(cache_d))

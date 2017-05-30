import math
import random
from collections import OrderedDict

import copy

import generator
import utils
from nmcs import dfs_advance_skeleton

from parsers import parse_typ, parse_ctx
from tree_node import UFTNode, ChooseKTNode, StackNode


class Environment:
    pass


def regression_domain_koza_poly_stack():
    def koza_poly(x):
        return x + x ** 2 + x ** 3 + x ** 4

    def raiserun():
        raise RuntimeError()

    symbols = {
        'plus': (2, (lambda x, y: x + y)),
        'minus': (2, (lambda x, y: x - y)),
        'times': (2, (lambda x, y: x * y)),
        'rdiv': (2, (lambda p, q: p / q if q else 1)),
        'rlog': (1, (lambda x: math.log(abs(x)) if x else 0)),
        'sin': (1, math.sin),
        'cos': (1, math.cos),
        'exp': (1, math.exp),
        'x': (0, raiserun),
    }

    all_symbols = list(symbols.keys())
    terminals = {k: v for k, v in symbols.items() if v[0] == 0}
    terminal_symbols = list(terminals.keys())

    cache_d = {}

    def count_missing(stack):
        missing = 1
        non_terminals = 0
        terminals = 0

        for symbol in stack:
            arity, _ = symbols.get(symbol, (0, None))
            missing += arity - 1
            if arity:
                non_terminals += 1
            else:
                terminals += 1
        return missing, terminals, non_terminals

    def finish(stack):
        nonfinished = count_missing(stack)[0]
        assert nonfinished >= 0
        if not nonfinished:
            return stack
        stack = copy.copy(stack)

        for _ in range(nonfinished):
            stack.append(random.choice(terminal_symbols))

        return stack

    def successors(stack, limit):
        missing, terminals, non_terminals = count_missing(stack)
        if not missing:
            return []
        if terminals + non_terminals >= limit:
            return [stack + [symbol] for symbol in terminal_symbols]

        return [stack + [symbol] for symbol in all_symbols]

    def fitness(stack, target_f=koza_poly, num_samples=1):
        assert count_missing(stack)[0] == 0

        def fun(val):
            s = copy.copy(stack)
            v = []

            while len(s) > 1 or len(v) > 0:
                print(s, v)
                symbol = s.pop()
                if symbol in symbols:
                    arity, fn = symbols[symbol]
                    if arity == 0 and symbol == 'x':
                        v.append(val)
                    else:
                        args = reversed(v[-arity:])
                        v[-arity:] = []
                        s.append(fn(*args))
                else:
                    v.append(symbol)

            assert len(s) == 1 and not v
            print(s, v)
            return s[0]

        s = repr(stack)
        samples = [-1 + 0.1 * i for i in range(num_samples)]
        try:
            error = sum(abs(fun(val) - target_f(val)) for val in samples)
        except OverflowError:
            return 0.0
        score = 1 / (1 + error)
        cache_d[s] = score

        return score

    return finish, successors, fitness


def make_env_stack():
    finish, successors, fitness = regression_domain_koza_poly_stack()

    env = Environment()

    #
    #  now define tree-searching functions in this env
    #

    @utils.pp_function('fitness()')
    def ffitness(node):
        assert isinstance(node, StackNode)
        return fitness(node.stack)

    @utils.pp_function('finish()')
    def ffinish(node):
        assert isinstance(node, StackNode)

        return StackNode(finish(node.stack))

    @utils.pp_function('successors()')
    def fsuccessors(node):
        assert isinstance(node, StackNode)
        return StackNode(successors(node.stack))

    @utils.pp_function('advance()')
    def fadvance(node, finished_node):
        assert isinstance(node, UFTNode) or isinstance(node, ChooseKTNode)
        assert isinstance(finished_node, UFTNode)
        new_uf_tree = dfs_advance_skeleton(node.uf_tree, finished_node.uf_tree)
        if new_uf_tree is None:
            return None
        return UFTNode(new_uf_tree, finished_node.k)

    env.t_fitness = ffitness
    env.t_finish = ffinish
    env.t_successors = fsuccessors
    env.t_advance = fadvance

    return env


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


def make_env():
    goal, gamma, fitness, count_evals = regression_domain_koza_poly()
    gen = generator.Generator(gamma)

    env = Environment()

    #
    #  now define tree-searching functions in this env
    #

    @utils.pp_function('fitness()')
    def ffitness(tree):
        assert isinstance(tree, UFTNode)
        # make sure we only run fitness on finished,
        # fully typed trees
        assert tree.uf_tree.typ is not None
        return fitness(tree.uf_tree)

    @utils.pp_function('finish()')
    def ffinish(tree):
        assert isinstance(tree, UFTNode)
        assert tree.uf_tree.typ is None

        finished_tree = gen.gen_one_uf(tree.uf_tree, tree.k, goal)
        assert finished_tree.typ is not None
        return UFTNode(finished_tree, tree.k)

    @utils.pp_function('successors()')
    def fsuccessors(tree):
        if isinstance(tree, UFTNode):
            return [UFTNode(c, tree.k) for c in tree.uf_tree.successors(gen, tree.k, goal)]
        if isinstance(tree, ChooseKTNode):
            return [UFTNode(tree.uf_tree, k) for k in range(2, tree.max_k + 1)]
        assert False

    @utils.pp_function('advance()')
    def fadvance(tree, finished_tree):
        assert isinstance(tree, UFTNode) or isinstance(tree, ChooseKTNode)
        assert isinstance(finished_tree, UFTNode)
        new_uf_tree = dfs_advance_skeleton(tree.uf_tree, finished_tree.uf_tree)
        if new_uf_tree is None:
            return None
        return UFTNode(new_uf_tree, finished_tree.k)

    env.t_fitness = ffitness
    env.t_finish = ffinish
    env.t_successors = fsuccessors
    env.t_advance = fadvance
    env.count_evals = count_evals

    return env


if __name__ == "__main__":
    finish, successors, fitness = regression_domain_koza_poly_stack()

    fitness(['times', 'plus', 4, 'x', 2])
    fitness(['rdiv', 'plus', 5, 'x', 2])

    print(finish(['rdiv', 'plus', 5, 'x', 2]))
    print(finish(['rdiv', 'plus', 2]))

    print(successors(['times', 'plus', 4, 'x'], 5))

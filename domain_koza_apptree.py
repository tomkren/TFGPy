import math
from collections import OrderedDict

import generator
import utils
from fitness_cache import FitnessCache
from nmcs import dfs_advance_skeleton
from parsers import parse_typ, parse_ctx
from tree_node import UFTNode, ChooseKTNode


class Environment:
    pass


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

    cache = FitnessCache()

    def fitness(individual_app_tree, target_f=koza_poly, num_samples=20):
        s = "lambda x : %s" % individual_app_tree.eval_str()
        cres = cache.d.get(s, None)
        if cres is not None:
            return cres

        fun = eval(s, global_symbols)
        assert callable(fun)
        samples = [-1 + 0.1 * i for i in range(num_samples)]
        try:
            error = sum(abs(fun(val) - target_f(val)) for val in samples)
        except (OverflowError, ValueError):
            return 0.0
        score = 1 / (1 + error)
        cache.update(s, score)
        return score

    return goal, gamma, fitness, (lambda: len(cache))


def make_env_app_tree():
    goal, gamma, raw_fitness, count_evals = regression_domain_koza_poly()
    gen = generator.Generator(gamma)

    env = Environment()

    #
    #  now define tree-searching functions in this env
    #

    @utils.pp_function('fitness()')
    def fitness(node):
        assert isinstance(node, UFTNode)
        # make sure we only run fitness on finished,
        # fully typed trees
        assert node.uf_tree.typ is not None
        return raw_fitness(node.uf_tree)

    @utils.pp_function('finish()')
    def finish(node):
        assert isinstance(node, UFTNode)
        if node.uf_tree.typ is not None:
            assert node.uf_tree.is_finished()
            return node.uf_tree

        finished_tree = gen.gen_one_uf(node.uf_tree, node.k, goal)
        assert finished_tree.typ is not None
        return UFTNode(finished_tree, node.k)

    @utils.pp_function('is_finished()')
    def is_finished(node):
        assert isinstance(node, UFTNode)
        return node.uf_tree.is_finished()

    @utils.pp_function('successors()')
    def successors(node):
        if isinstance(node, ChooseKTNode):
            return [UFTNode(node.uf_tree, k) for k in range(2, node.max_k + 1)]
        if isinstance(node, UFTNode):
            return [UFTNode(c, node.k) for c in node.uf_tree.successors(gen, node.k, goal)]
        assert False

    @utils.pp_function('advance()')
    def advance(node, finished_tree):
        assert isinstance(node, UFTNode) or isinstance(node, ChooseKTNode)
        assert isinstance(finished_tree, UFTNode)
        new_uf_tree = dfs_advance_skeleton(node.uf_tree, finished_tree.uf_tree)
        if new_uf_tree is None:
            return None
        return UFTNode(new_uf_tree, finished_tree.k)

    env.fitness = fitness
    env.finish = finish
    env.is_finished = is_finished
    env.successors = successors
    env.advance = advance
    env.count_evals = count_evals

    return env

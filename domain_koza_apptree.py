import math
from collections import OrderedDict

import app_tree
import generator
import utils
from fitness_cache import FitnessCache
from nmcs import dfs_advance_skeleton
from parsers import parse_typ, parse_ctx
from tree_node import UFTNode, ChooseKTNode, MaxKTNode

size_d = {}


class Environment:
    pass


def koza_poly(x):
    return x + x ** 2 + x ** 3 + x ** 4


def eval_regression(fun, target_f, num_samples=20):
    assert callable(fun)
    samples = [-1 + 0.1 * i for i in range(num_samples)]
    try:
        error = sum(abs(fun(val) - target_f(val)) for val in samples)
        score = 1 / (1 + error)
    except (OverflowError, ValueError):
        score = 0.0
    return score


def d():
    R = 'R'
    return (parse_typ(R),
            parse_ctx(OrderedDict([
                ('plus', (R, '->', (R, '->', R))),
                ('minus', (R, '->', (R, '->', R))),
                ('times', (R, '->', (R, '->', R))),
                ('rdiv', (R, '->', (R, '->', R))),
                ('sin', (R, '->', R)),
                ('cos', (R, '->', R)),
                ('exp', (R, '->', R)),
                ('rlog', (R, '->', R)),
                ('x', R),
            ])), 20)


def regression_domain_koza():
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

    goal, gamma, _ = d()
    cache = FitnessCache()

    def fitness(individual_app_tree):
        global size_d
        size = individual_app_tree.count_nodes()[app_tree.Leaf]
        size_d[size] = size_d.get(size, 0) + 1

        s = "lambda x : %s" % individual_app_tree.eval_str()
        cres = cache.d.get(s, None)
        if cres is not None:
            return cres
        fun = eval(s, global_symbols)

        score = eval_regression(fun, koza_poly, 20)

        cache.update(s, score)
        return score

    return goal, gamma, fitness, (lambda: len(cache)), cache


def make_env_app_tree(get_raw_domain=regression_domain_koza, early_end_limit=1.0, smart=False):
    goal, gamma, raw_fitness, count_evals, cache = get_raw_domain()

    if smart:
        gen = generator.GeneratorSmart(gamma)
    else:
        gen = generator.Generator(gamma)

    env = Environment()
    env.cache = cache

    #
    #  now define tree-searching functions in this env
    #

    @utils.pp_function('fitness()')
    def fitness(node):
        assert isinstance(node, UFTNode)
        # make sure we only run fitness on finished,
        # fully typed trees
        if smart:
            assert node.uf_tree.is_finished()
        else:
            assert node.uf_tree.typ is not None
        return raw_fitness(node.uf_tree)

    @utils.pp_function('early_end_test()')
    def early_end_test(score):
        return early_end_limit is None or score >= early_end_limit

    @utils.pp_function('finish()')
    def finish(node):
        assert isinstance(node, UFTNode)

        if smart:
            if node.uf_tree.is_finished():
                return node
        else:
            if node.uf_tree.typ is not None:
                assert False
                # sem to nikdy nedojde???
                assert node.uf_tree.is_finished()
                return node

        # TODO this branch is not done
        # it is possible to do the same as in MacxKTNode
        assert not isinstance(node, ChooseKTNode)

        if isinstance(node, MaxKTNode):
            finished = gen.gen_one_uf_up_to(node.uf_tree, node.max_k, goal)
            assert finished is not None
            finished_tree, k = finished
            ret = UFTNode(finished_tree, k)
        elif isinstance(node, UFTNode):
            finished_tree = gen.gen_one_uf(node.uf_tree, node.k, goal)
            ret = UFTNode(finished_tree, node.k)
        else:
            assert False

        # this means that the node we are to finish
        # is not populated (no such tree exists)
        # in such case, we should not even be called
        assert ret.uf_tree is not None
        if smart:
            assert ret.uf_tree.is_finished()
        else:
            assert ret.uf_tree.typ is not None
        return ret

    @utils.pp_function('is_finished()')
    def is_finished(node):
        assert isinstance(node, UFTNode)
        return node.uf_tree.is_finished()

    @utils.pp_function('successors()')
    def successors(node):
        if isinstance(node, ChooseKTNode):
            return [UFTNode(node.uf_tree, k) for k in range(2, node.max_k + 1)
                    # if gen.get_num_uf(node.uf_tree, k, goal)
                    ]
        if isinstance(node, MaxKTNode):
            ret = [MaxKTNode(c, node.max_k) for c in gen.uf_tree_successors_up_to(node.uf_tree, node.max_k, goal)]
            return ret
        if isinstance(node, UFTNode):
            return [UFTNode(c, node.k) for c in gen.uf_tree_successors(node.uf_tree, node.k, goal)]
        assert False

    @utils.pp_function('advance()')
    def advance(node, finished_tree):
        assert isinstance(node, UFTNode) or isinstance(node, ChooseKTNode)
        assert isinstance(finished_tree, UFTNode)
        new_uf_tree = dfs_advance_skeleton(node.uf_tree, finished_tree.uf_tree)
        if new_uf_tree is None:
            return None
        if isinstance(node, MaxKTNode):
            return MaxKTNode(new_uf_tree, node.max_k)
        if isinstance(node, UFTNode):
            return UFTNode(new_uf_tree, finished_tree.k)

    env.goal = goal
    env.fitness = fitness
    env.early_end_test = early_end_test
    env.finish = finish
    env.is_finished = is_finished
    env.successors = successors
    env.advance = advance
    env.count_evals = count_evals

    return env

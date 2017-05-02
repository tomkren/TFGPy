import random
from collections import OrderedDict

import generator
from app_tree import App, UnfinishedLeaf, UNFINISHED_APP
from app_tree import Leaf
from parsers import parse_ctx, parse_typ


def dfs_skeleton_picker(tree, old_skeleton):
    if isinstance(old_skeleton, App):
        assert isinstance(tree, App)
        maybe_new_fun = dfs_skeleton_picker(tree.fun, old_skeleton.fun)
        if maybe_new_fun is not None:
            return App(maybe_new_fun, old_skeleton.arg, None)

        maybe_new_arg = dfs_skeleton_picker(tree.arg, old_skeleton.arg)
        if maybe_new_arg is not None:
            return App(old_skeleton.fun, maybe_new_arg)

        return None

    if isinstance(old_skeleton, Leaf):
        return None

    if isinstance(old_skeleton, UnfinishedLeaf):
        if isinstance(tree, Leaf):
            return tree
        if isinstance(tree, App):
            return UNFINISHED_APP
        assert False

    assert False


def nested_mc_search(uf_tree, level, gen, eval_score, k, typ, skeleton_picker_strategy):
    if level == 0:
        tree = gen.gen_one_uf(uf_tree, k, typ)
        tree.score = eval_score(tree)
        return tree

    best = None
    while uf_tree.is_unfinished():
        for uf_successor in uf_tree.successors(gen, k, typ):
            tree = nested_mc_search(uf_successor, level - 1, gen, eval_score, k, typ)
            if best is None:
                best = tree
            else:
                best = tree if tree.score > best.score else best
        assert best is not None
        assert not best.is_unfinished()

        assert uf_tree.is_skeleton_of(best)
        uf_tree_new = skeleton_picker_strategy(best, uf_tree)
        assert uf_tree_new is not None
        assert uf_tree_new.is_skeleton_of(best)
        assert uf_tree.count_symbols() + 1 == uf_tree_new.count_symbols()
        uf_tree = uf_tree_new

    return best


if __name__ == "__main__":
    import math


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


    def make_fun(individual):
        s = "lambda x : %s" % individual.eval_str()
        f = eval(s, global_symbols)
        assert callable(f)
        return f


    def fitness(individual, target_f=koza_poly, num_samples=20):
        fun = make_fun(individual)
        samples = [-1 + 0.1 * i for i in range(num_samples)]
        return sum(abs(fun(val) - target_f(val)) for val in samples)


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
        ('x', R)
    ]))

    gen = generator.Generator(gamma)
    random.seed(5)
    indiv = gen.gen_one(30, goal)
    print(indiv.eval_str())
    print(fitness(indiv))

    # best = nested_mc_search(uf_tree, level, gen, eval_score, k, typ, skeleton_picker_strategy):

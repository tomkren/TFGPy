import random
from collections import OrderedDict

import generator
from app_tree import App, UnfinishedLeaf, UNFINISHED_APP
from app_tree import Leaf
from parsers import parse_ctx, parse_typ
from tracer_deco import tracer_deco


def dfs_skeleton_picker(tree, old_skeleton):
    if isinstance(old_skeleton, App):
        assert isinstance(tree, App)
        maybe_new_fun = dfs_skeleton_picker(tree.fun, old_skeleton.fun)
        if maybe_new_fun is not None:
            return App(maybe_new_fun, old_skeleton.arg, None)

        maybe_new_arg = dfs_skeleton_picker(tree.arg, old_skeleton.arg)
        if maybe_new_arg is not None:
            return App(old_skeleton.fun, maybe_new_arg, None)

        return None

    if isinstance(old_skeleton, UnfinishedLeaf):
        if isinstance(tree, Leaf):
            return tree
        if isinstance(tree, App):
            return UNFINISHED_APP
        assert False

    if isinstance(old_skeleton, Leaf):
        return None

    assert False


def nested_mc_search(gen, goal, max_k, eval_score, max_level, skeleton_picker_strategy=None):
    if skeleton_picker_strategy is None:
        skeleton_picker_strategy = dfs_skeleton_picker

    best = None
    for k in range(1, max_k+1):
        print("="*10,"k=", k)
        this = nested_mc_search_raw(UnfinishedLeaf(), max_level, gen, eval_score, k, goal, skeleton_picker_strategy)
        print(this)
        if best is None or this.score > best.score:
            best = this

    return best


#@tracer_deco(log_ret=True, enable=True)
def nested_mc_search_raw(uf_tree, level, gen, eval_score, k, typ, skeleton_picker_strategy):
    print(level, uf_tree)
    if level == 0:
        tree = gen.gen_one_uf(uf_tree, k, typ)
        tree.score = eval_score(tree)
        return tree

    best = None
    # None symbolizes finished tree with no UnfinishedLeaves
    while uf_tree is not None:
        for uf_successor in uf_tree.successors(gen, k, typ):
            #print("SUCC:", uf_successor)
            tree = nested_mc_search_raw(uf_successor, level - 1, gen, eval_score, k, typ, skeleton_picker_strategy)
            if best is None or tree.score > best.score:
                best = tree

        assert best is not None
        assert not best.is_unfinished()

        assert uf_tree.is_skeleton_of(best)
        uf_tree_new = skeleton_picker_strategy(best, uf_tree)
        if uf_tree_new is not None:
            assert uf_tree_new.is_skeleton_of(best)
            before = uf_tree.count_finished_nodes()
            after = uf_tree_new.count_finished_nodes()
            assert before + 1 == after
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
        try:
            error = sum(abs(fun(val) - target_f(val)) for val in samples)
        except OverflowError:
            return 0.0
        score = 1 / (1+error)

        print("EVAL", individual, score)
        return score


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
    if False:
        random.seed(5)
        indiv = gen.gen_one(30, goal)
        print(indiv.eval_str())
        print(fitness(indiv))

    for i in range(1):
        print("="*10, i, "="*10)
        indiv = nested_mc_search(gen, goal, 10, fitness, 1)
        print(indiv.eval_str())
        print(fitness(indiv))






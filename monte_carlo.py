import random
from collections import OrderedDict

import generator
from app_tree import App, UnfinishedLeaf, UNFINISHED_APP
from app_tree import Leaf
from parsers import parse_ctx, parse_typ
from tracer_deco import tracer_deco


def dfs_advance_skeleton(old_skeleton, finished_tree):
    """Advance `old_skeleton` one step towards the `finished_tree`."""
    assert finished_tree.is_finished()
    assert old_skeleton.is_skeleton_of(finished_tree)

    if isinstance(old_skeleton, App):
        assert isinstance(finished_tree, App)
        maybe_new_fun = dfs_advance_skeleton(old_skeleton.fun, finished_tree.fun)
        if maybe_new_fun is not None:
            return App(maybe_new_fun, old_skeleton.arg)

        maybe_new_arg = dfs_advance_skeleton(old_skeleton.arg, finished_tree.arg)
        if maybe_new_arg is not None:
            return App(old_skeleton.fun, maybe_new_arg)
        return None

    if isinstance(old_skeleton, UnfinishedLeaf):
        if isinstance(finished_tree, Leaf):
            return finished_tree
        if isinstance(finished_tree, App):
            return UNFINISHED_APP
        assert False

    if isinstance(old_skeleton, Leaf):
        return None

    assert False


def nested_mc_search(repeat, gen, goal_typ, max_k, max_level, fitness, advance_skeleton=None):
    #@tracer_deco(print_from_arg=0)
    def nested_mc_search_raw(level, k, uf_tree):
        # print(k, level, uf_tree, flush=True)

        # if the input tree is already finished, evaluate it
        if uf_tree.is_finished():
            uf_tree.score = fitness(uf_tree)
            return uf_tree

        # on level 0, finish the tree uniformly randomly with size k and evaluate
        if level == 0:
            tree = gen.gen_one_uf(uf_tree, k, goal_typ)
            tree.score = fitness(tree)
            return tree

        # best finished tree found
        best_tree = None
        while uf_tree is not None:
            for uf_successor in uf_tree.successors(gen, k, goal_typ):
                # print("SUCC:", uf_successor)
                tree = nested_mc_search_raw(level - 1, k, uf_successor)
                if best_tree is None or tree.score > best_tree.score:
                    best_tree = tree
                    # print("BEST NEMC", best.score)
            # the successors must be nonempty, so:
            assert best_tree is not None

            uf_tree_new = advance_skeleton(uf_tree, best_tree)
            # advance_skeleton should return None
            # when the uf_tree (old skeleton) is already finished
            if uf_tree_new is not None:
                assert uf_tree_new.is_skeleton_of(best_tree)
                # the advance_skeleton should concretize just one symbol
                assert uf_tree.count_finished_nodes() + 1 == uf_tree_new.count_finished_nodes()
            uf_tree = uf_tree_new

        return best_tree

    if advance_skeleton is None:
        advance_skeleton = dfs_advance_skeleton

    best = None
    for k in range(1, max_k + 1):
        print("=" * 10, "k=", k)
        this = nested_mc_search_raw(max_level, k, UnfinishedLeaf())
        # print(this)
        if best is None or this.score > best.score:
            best = this
            print("BEST K", best.score)

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

    CACHE = {}


    def fitness(individual, target_f=koza_poly, num_samples=20):
        s = "lambda x : %s" % individual.eval_str()
        if s in CACHE:
            return CACHE[s]

        fun = eval(s, global_symbols)
        assert callable(fun)
        samples = [-1 + 0.1 * i for i in range(num_samples)]
        try:
            error = sum(abs(fun(val) - target_f(val)) for val in samples)
        except OverflowError:
            return 0.0
        score = 1 / (1 + error)
        CACHE[s] = score

        # print("EVAL", individual, score)
        return score


    gen = generator.Generator(gamma)
    if False:
        random.seed(5)
        indiv = gen.gen_one(30, goal)
        print(indiv.eval_str())
        print(fitness(indiv))

    fs = []
    for i in range(1):
        print("=" * 10, i, "=" * 10)
        indiv = nested_mc_search(1, gen, goal, max_k=5, max_level=3, fitness=fitness)
        print(indiv.eval_str())
        fi = fitness(indiv)
        fs.append(fi)
        print(fi)

    print()
    print(sum(fs) / len(fs), min(fs), max(fs))
    print(len(CACHE))

import copy
import math
import random

import utils
from domain_koza_apptree import Environment, koza_poly, eval_regression
from fitness_cache import FitnessCache
from stack import Stack
from tree_node import StackNode

size_d = {}


def regression_domain_koza_stack():
    def raiserun():
        raise RuntimeError()

    symbols_d = {
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
    sutils = Stack(symbols_d)

    cache = FitnessCache()

    def fitness(stack):
        global size_d
        size_d[len(stack)] = size_d.get(len(stack), 0) + 1

        assert sutils.count_missing(stack)[0] == 0
        s = repr(stack)

        cres = cache.d.get(s, None)
        if cres is not None:
            return cres

        fun = lambda val: sutils.eval_stack(stack, {'x': val})
        score = eval_regression(fun, koza_poly, 20)

        cache.update(s, score)
        return score

    return sutils.finish, sutils.is_finished, sutils.successors, fitness, sutils.eval_stack, (lambda: len(cache)), cache


def make_env_stack(max_k=5, get_raw_domain=regression_domain_koza_stack, early_end_limit=1.0):
    raw_finish, raw_is_finished, raw_successors, raw_fitness, raw_eval, count_evals, cache = get_raw_domain()
    env = Environment()
    env.count_evals = count_evals
    env.cache = cache

    #
    #  now define tree-searching functions in this env
    #

    @utils.pp_function('fitness()')
    def fitness(node):
        assert isinstance(node, StackNode)
        return raw_fitness(node.stack)

    @utils.pp_function('early_end_test()')
    def early_end_test(score):
        return score >= early_end_limit

    @utils.pp_function('is_finished()')
    def is_finished(node):
        assert isinstance(node, StackNode)
        return raw_is_finished(node.stack)

    @utils.pp_function('finish()')
    def finish(node):
        assert isinstance(node, StackNode)

        return StackNode(raw_finish(node.stack, max_k))

    @utils.pp_function('successors()')
    def successors(node):
        assert isinstance(node, StackNode)
        return [StackNode(stack) for stack in raw_successors(node.stack, max_k)]

    @utils.pp_function('advance()')
    def advance(node, finished_node):
        assert isinstance(node, StackNode)
        assert isinstance(finished_node, StackNode)
        assert is_finished(finished_node)

        f_stack, stack = finished_node.stack, node.stack

        assert stack == f_stack[:len(stack)]
        if len(f_stack) <= len(stack):
            assert f_stack == stack
            return None

        stack = copy.copy(stack)
        stack.append(f_stack[len(stack)])
        return StackNode(stack)

    env.fitness = fitness
    env.early_end_test = early_end_test
    env.finish = finish
    env.is_finished = is_finished
    env.successors = successors
    env.advance = advance

    return env


if __name__ == "__main__":
    finish, is_finished, successors, fitness, eval_stack, count_evals, cache = regression_domain_koza_stack()

    for i in range(10000):
        s = ['plus', 'plus']
        f = finish(s, 15)
        fitness(f)

    print('\n'.join("%s = %s" % (a, b) for a, b in sorted(size_d.items())))

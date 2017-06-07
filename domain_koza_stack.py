import copy
import math
import random

import utils
from domain_koza_apptree import Environment
from fitness_cache import FitnessCache
from tree_node import StackNode


def regression_domain_koza_poly_stack():
    def koza_poly(x):
        return x + x ** 2 + x ** 3 + x ** 4

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

    all_symbols = list(symbols_d.keys())
    terminals_d = {k: v for k, v in symbols_d.items() if v[0] == 0}
    terminal_symbols = list(terminals_d.keys())

    cache = FitnessCache()

    def count_missing(stack):
        missing = 1
        non_terminals = 0
        terminals = 0

        for symbol in stack:
            arity, _ = symbols_d.get(symbol, (0, None))
            missing += arity - 1
            if arity:
                non_terminals += 1
            else:
                terminals += 1
        return missing, terminals, non_terminals

    def successor_symbols(stack, limit, count=None):
        missing, terminals, non_terminals = count if count is not None else count_missing(stack)

        if not missing:
            return []
        if terminals + non_terminals + missing >= limit:
            return terminal_symbols
        return all_symbols

    def finish(stack, limit):
        missing, terminals, non_terminals = count_missing(stack)

        assert missing >= 0
        if not missing:
            return stack
        stack = copy.copy(stack)

        while missing > 0:
            ssym = successor_symbols(stack, limit, (missing, terminals, non_terminals))
            assert ssym
            ns = random.choice(ssym)

            arity, _ = symbols_d[ns]
            if not arity:
                missing -= 1
                terminals += 1
            else:
                missing = missing - 1 + arity
                non_terminals += 1

            stack.append(ns)

        return stack

    def is_finished(stack):
        nonfinished = count_missing(stack)[0]
        return nonfinished == 0

    def successors(stack, limit):
        count = count_missing(stack)
        ssym = successor_symbols(stack, limit, count)

        if not ssym:
            return []

        return [stack + [s] for s in ssym]

    def eval_stack(stack, val):
        s = copy.copy(stack)
        v = []

        while len(s):
            # print(s, v)
            symbol = s.pop()
            if symbol in symbols_d:
                arity, fn = symbols_d[symbol]
                if arity == 0 and symbol == 'x':
                    v.append(val)
                else:
                    args = reversed(v[-arity:])
                    v[-arity:] = []
                    v.append(fn(*args))
            else:
                v.append(symbol)

        assert len(v) == 1 and not s
        # print(s, v)
        return v[0]

    def fitness(stack, target_f=koza_poly, num_samples=20):
        assert count_missing(stack)[0] == 0
        s = repr(stack)

        cres = cache.d.get(s, None)
        if cres is not None:
            return cres

        samples = [-1 + 0.1 * i for i in range(num_samples)]
        try:
            error = 0
            for val in samples:
                fv = eval_stack(stack, val)
                tv = target_f(val)
                error += abs(fv - tv)
        except (OverflowError, ValueError):
            return 0.0
        score = 1 / (1 + error)

        cache.update(s, score)
        return score

    return finish, is_finished, successors, fitness, eval_stack, (lambda: len(cache))


def make_env_stack(limit=5):
    raw_finish, raw_is_finished, raw_successors, raw_fitness, raw_eval, count_evals = regression_domain_koza_poly_stack()
    env = Environment()
    env.count_evals = count_evals

    #
    #  now define tree-searching functions in this env
    #

    @utils.pp_function('fitness()')
    def fitness(node):
        assert isinstance(node, StackNode)
        return raw_fitness(node.stack)

    @utils.pp_function('is_finished()')
    def is_finished(node):
        assert isinstance(node, StackNode)
        return raw_is_finished(node.stack)

    @utils.pp_function('finish()')
    def finish(node):
        assert isinstance(node, StackNode)

        return StackNode(raw_finish(node.stack, limit))

    @utils.pp_function('successors()')
    def successors(node):
        assert isinstance(node, StackNode)
        return [StackNode(stack) for stack in raw_successors(node.stack, limit)]

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
    env.finish = finish
    env.is_finished = is_finished
    env.successors = successors
    env.advance = advance

    return env


if __name__ == "__main__":
    finish, is_finished, successors, fitness, eval_stack, count_evals = regression_domain_koza_poly_stack()

    print(eval_stack(['x'], 2))
    print(successors(['times', 'plus', 4, 'x'], 5))

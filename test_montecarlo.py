import math
import random
import unittest
from collections import OrderedDict

import copy

import generator
import utils
from app_tree import UnfinishedLeaf
from mcts import MCTNode
from mcts import mct_search
from nmcs import dfs_advance_skeleton, nested_mc_search

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

    def is_finished(stack):
        nonfinished = count_missing(stack)[0]
        return nonfinished == 0

    def successors(stack, limit):
        missing, terminals, non_terminals = count_missing(stack)
        if not missing:
            return []
        if terminals + non_terminals >= limit:
            return [stack + [symbol] for symbol in terminal_symbols]

        return [stack + [symbol] for symbol in all_symbols]

    def eval_stack(stack, val):
        s = copy.copy(stack)
        v = []

        while len(s):
            #print(s, v)
            symbol = s.pop()
            if symbol in symbols:
                arity, fn = symbols[symbol]
                if arity == 0 and symbol == 'x':
                    v.append(val)
                else:
                    args = reversed(v[-arity:])
                    v[-arity:] = []
                    v.append(fn(*args))
            else:
                v.append(symbol)

        assert len(v) == 1 and not s
        #print(s, v)
        return v[0]

    def fitness(stack, target_f=koza_poly, num_samples=20):
        assert count_missing(stack)[0] == 0

        s = repr(stack)
        samples = [-1 + 0.1 * i for i in range(num_samples)]
        try:
            error = 0
            for val in samples:
                fv = eval_stack(stack, val)
                tv = target_f(val)
                error += abs(fv - tv)
        except OverflowError:
            return 0.0
        score = 1 / (1 + error)
        cache_d[s] = score

        return score

    return finish, is_finished, successors, fitness, eval_stack, (lambda: len(cache_d))


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

        return StackNode(raw_finish(node.stack))

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


class TestKozaRegressionDomain(unittest.TestCase):
    def test_domain(self):
        goal, gamma, raw_fitness, count_evals = regression_domain_koza_poly()
        gen = generator.Generator(gamma)
        random.seed(5)
        indiv = gen.gen_one(20, goal)
        istr = indiv.eval_str()
        ifit = raw_fitness(indiv)
        if False:
            print(istr)
            print(ifit)
        self.assertTrue(True)


class TestStack(unittest.TestCase):
    def test(self):
        finish, is_finished, successors, fitness, eval_stack, count_evals = regression_domain_koza_poly_stack()

        self.assertEqual(eval_stack(['x'], 3), 3)
        self.assertEqual(eval_stack(['plus', 6, 'x'], 3), 6 + 3)
        self.assertEqual(eval_stack(['times', 6, 'x'], 3), 6 * 3)
        self.assertEqual(eval_stack(['rdiv', 'plus', 5, 'x', 2], 3), (5 + 3) / 2)

        a = ['rdiv', 'plus', 5, 'x', 2]
        self.assertEqual(finish(a), a)
        b = ['rdiv', 'plus', 2]
        self.assertEqual(finish(b), b + ['x', 'x'])


class TestMCRegression(unittest.TestCase):
    def test_nmcs(self):
        env = make_env()
        nested_mc_search(ChooseKTNode(UnfinishedLeaf(), 5),
                         max_level=1,
                         fitness=env.fitness,
                         finish=env.finish,
                         is_finished=env.is_finished,
                         successors=env.successors,
                         advance=env.advance)
        # the test just checks that nothing dies with exception
        self.assertTrue(True)

    def test_mcts(self):
        env = make_env()
        root = MCTNode(ChooseKTNode(UnfinishedLeaf(), 5))
        mct_search(root, expand_visits=1, num_steps=50,
                   fitness=env.fitness,
                   finish=env.finish,
                   is_finished=env.is_finished,
                   successors=env.successors)
        # the test just checks that nothing dies with exception
        self.assertTrue(True)


class TestMCRegressionStack(unittest.TestCase):
    def test_nmcs(self):
        env = make_env_stack(5)
        nested_mc_search(StackNode([]),
                         max_level=1,
                         fitness=env.fitness,
                         finish=env.finish,
                         is_finished=env.is_finished,
                         successors=env.successors,
                         advance=env.advance)
        # the test just checks that nothing dies with exception
        self.assertTrue(True)

    def test_mcts(self):
        env = make_env_stack(5)
        root = MCTNode(StackNode([]))
        mct_search(root, expand_visits=1, num_steps=50,
                   fitness=env.fitness,
                   finish=env.finish,
                   is_finished=env.is_finished,
                   successors=env.successors)
        # the test just checks that nothing dies with exception
        self.assertTrue(True)


if __name__ == "__main__":
    if True:
        unittest.main()
    else:
        finish, is_finished, successors, fitness, eval_stack, count_evals = regression_domain_koza_poly_stack()

        print(eval_stack(['x'], 2))

        # print(successors(['times', 'plus', 4, 'x'], 5))

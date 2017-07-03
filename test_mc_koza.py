import random
import unittest

import domain_koza_apptree
import domain_koza_stack
import generator
from app_tree import UnfinishedLeaf
from mcts import MCTNode
from mcts import mct_search
from nmcs import nested_mc_search
from tree_node import ChooseKTNode, StackNode


def run_basic_nmcs(node, env, max_level=1):
    nested_mc_search(node,
                     max_level=max_level,
                     fitness=env.fitness,
                     finish=env.finish,
                     is_finished=env.is_finished,
                     successors=env.successors,
                     advance=env.advance)


def run_basic_mcts(node, env, expand=1, num_steps=10):
    mct_search(node, expand_visits=expand, num_steps=num_steps,
               fitness=env.fitness,
               finish=env.finish,
               is_finished=env.is_finished,
               successors=env.successors)


def run_gen_basic(domain_raw, size, verbose=False):
    goal, gamma, raw_fitness, count_evals, cache = domain_raw()
    gen = generator.Generator(gamma)
    random.seed(5)
    indiv = gen.gen_one(size, goal)
    assert indiv is not None
    istr = indiv.eval_str()
    ifit = raw_fitness(indiv)
    if verbose:
        print(istr)
        print(ifit)


class TestKozaRegressionDomainApptree(unittest.TestCase):
    def test_domain(self):
        run_gen_basic(domain_koza_apptree.regression_domain_koza, 20)


class TestKozaRegressionDomainStack(unittest.TestCase):
    def test(self):
        finish, is_finished, successors, fitness, eval_stack, count_evals, cache = domain_koza_stack.regression_domain_koza_stack()

        self.assertEqual(eval_stack(['x'], {'x': 3}), 3)
        self.assertEqual(eval_stack(['plus', 6, 'x'], {'x': 3}), 6 + 3)
        self.assertEqual(eval_stack(['times', 6, 'x'], {'x': 3}), 6 * 3)
        self.assertEqual(eval_stack(['rdiv', 'plus', 5, 'x', 2], {'x': 3}), (5 + 3) / 2)

        a = ['rdiv', 'plus', 5, 'x', 2]
        self.assertEqual(finish(a, 5), a)
        b = ['rdiv', 'plus', 2]
        self.assertEqual(finish(b, 5), b + ['x', 'x'])


class TestMCKozaRegressionApptree(unittest.TestCase):
    def test_nmcs(self):
        env = domain_koza_apptree.make_env_app_tree()
        node = ChooseKTNode(UnfinishedLeaf(), 10)
        run_basic_nmcs(node, env)

    def test_mcts(self):
        env = domain_koza_apptree.make_env_app_tree()
        node = MCTNode(ChooseKTNode(UnfinishedLeaf(), 5))
        run_basic_mcts(node, env)


class TestMCKozaRegressionStack(unittest.TestCase):
    def test_nmcs(self):
        env = domain_koza_stack.make_env_stack(10)
        node = StackNode([])
        run_basic_nmcs(node, env)

    def test_mcts(self):
        env = domain_koza_stack.make_env_stack(5)
        node = MCTNode(StackNode([]))
        run_basic_mcts(node, env)


if __name__ == "__main__":
    unittest.main()

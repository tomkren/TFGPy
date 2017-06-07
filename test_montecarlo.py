import random
import unittest

import generator
from app_tree import UnfinishedLeaf
from domain_koza_apptree import regression_domain_koza_poly, make_env_app_tree
from domain_koza_stack import regression_domain_koza_poly_stack, make_env_stack
from mcts import MCTNode
from mcts import mct_search
from nmcs import nested_mc_search
from tree_node import ChooseKTNode, StackNode


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
        self.assertEqual(finish(a, 5), a)
        b = ['rdiv', 'plus', 2]
        self.assertEqual(finish(b, 3), b + ['x', 'x'])


class TestMCRegression(unittest.TestCase):
    def test_nmcs(self):
        env = make_env_app_tree()
        nested_mc_search(ChooseKTNode(UnfinishedLeaf(), 10),
                         max_level=1,
                         fitness=env.fitness,
                         finish=env.finish,
                         is_finished=env.is_finished,
                         successors=env.successors,
                         advance=env.advance)
        # the test just checks that nothing dies with exception
        self.assertTrue(True)

    def test_mcts(self):
        env = make_env_app_tree()
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
        env = make_env_stack(10)
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
    unittest.main()

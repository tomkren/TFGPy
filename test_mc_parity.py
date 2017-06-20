import random
import unittest

import domain_parity_apptree
import domain_primes_apptree
import generator
from app_tree import UnfinishedLeaf
from mcts import MCTNode, mct_search
from nmcs import nested_mc_search
from tree_node import ChooseKTNode, MaxKTNode, UFTNode


class TestParityDomainApptree(unittest.TestCase):
    def test_domain(self):
        goal, gamma, raw_fitness, count_evals, cache = domain_parity_apptree.domain_parity(2)
        gen = generator.Generator(gamma)
        random.seed(5)
        indiv = gen.gen_one(17, goal)
        self.assertIsNotNone(indiv)
        istr = indiv.eval_str()
        ifit = raw_fitness(indiv)
        if False:
            print(istr)
            print(ifit)
        self.assertTrue(True)


class TestMCParityApptree(unittest.TestCase):
    def test_nmcs(self):
        env = domain_parity_apptree.make_env_app_tree()
        nested_mc_search(UFTNode(UnfinishedLeaf(), 3),
                         max_level=0,
                         fitness=env.fitness,
                         finish=env.finish,
                         is_finished=env.is_finished,
                         successors=env.successors,
                         advance=env.advance)
        # the test just checks that nothing dies with exception
        self.assertTrue(True)

    def test_mcts(self):
        env = domain_parity_apptree.make_env_app_tree()
        root = MCTNode(UFTNode(UnfinishedLeaf(), 3))
        mct_search(root, expand_visits=1, num_steps=3,
                   fitness=env.fitness,
                   finish=env.finish,
                   is_finished=env.is_finished,
                   successors=env.successors)
        # the test just checks that nothing dies with exception
        self.assertTrue(True)


if __name__ == "__main__":
    pass
    unittest.main()

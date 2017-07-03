import unittest

import domain_parity_apptree
import domain_parity_stack
from app_tree import UnfinishedLeaf
from mcts import MCTNode
from test_mc_koza import run_gen_basic, run_basic_nmcs, run_basic_mcts
from tree_node import UFTNode, StackNode


class TestParityDomainApptree(unittest.TestCase):
    def test_domain(self):
        run_gen_basic(lambda: domain_parity_apptree.domain_parity(6), 17)


class TestKozaRegressionDomainStack(unittest.TestCase):
    def test(self):
        self.assertTrue(True)


class TestMCParityStack(unittest.TestCase):
    def test_nmcs(self):
        env = domain_parity_stack.make_env_stack()
        node = StackNode([])
        run_basic_nmcs(node, env, 0)

    def test_mcts(self):
        env = domain_parity_stack.make_env_stack()
        node = MCTNode(StackNode([]))
        run_basic_mcts(node, env, 1, 3)


class TestMCParityApptree(unittest.TestCase):
    def test_nmcs(self):
        env = domain_parity_apptree.make_env_app_tree()
        node = UFTNode(UnfinishedLeaf(), 3)
        run_basic_nmcs(node, env, 0)

    def test_mcts(self):
        env = domain_parity_apptree.make_env_app_tree()
        node = MCTNode(UFTNode(UnfinishedLeaf(), 3))
        run_basic_mcts(node, env, 1, 3)


if __name__ == "__main__":
    pass
    unittest.main()

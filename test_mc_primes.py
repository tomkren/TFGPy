import unittest

import domain_primes_apptree
from app_tree import UnfinishedLeaf
from mcts import MCTNode
from test_mc_koza import run_gen_basic, run_basic_mcts, run_basic_nmcs
from tree_node import UFTNode


class TestPrimesDomainApptree(unittest.TestCase):
    def test_domain(self):
        run_gen_basic(domain_primes_apptree.domain_primes, 17)


class TestMCPrimesApptree(unittest.TestCase):
    def test_nmcs(self):
        env = domain_primes_apptree.make_env_app_tree()
        node = UFTNode(UnfinishedLeaf(), 3)
        run_basic_nmcs(node, env, 0)

    def test_mcts(self):
        env = domain_primes_apptree.make_env_app_tree()
        node = MCTNode(UFTNode(UnfinishedLeaf(), 3))
        run_basic_mcts(node, env, 1, 3)


if __name__ == "__main__":
    pass
    unittest.main()

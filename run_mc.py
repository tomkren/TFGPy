import time

from app_tree import UnfinishedLeaf
from mcts import MCTNode, mct_search
from nmcs import nested_mc_search
from domain_koza_apptree import make_env_app_tree
from domain_koza_stack import make_env_stack
from tree_node import ChooseKTNode, StackNode
from tree_stats import TreeStats
from utils import experiment_eval

if __name__ == "__main__":
    make_env = make_env_app_tree
    #make_env = lambda: make_env_stack(20)

    if False:
        # MCTS - one run
        env = make_env()
        # tracer_deco.enable_tracer = True
        # random.seed(5)
        tree_stats = TreeStats()

        # root = MCTNode(ChooseKTNode(UnfinishedLeaf(), 5))
        root = MCTNode(StackNode([]))
        mct_search(root, expand_visits=8, num_steps=4000,
                   fitness=env.fitness,
                   finish=env.finish,
                   is_finished=env.is_finished,
                   successors=env.successors,
                   tree_stats=tree_stats)
        print('=' * 20)
        print(root.pretty_str())
        print(tree_stats.pretty_str())

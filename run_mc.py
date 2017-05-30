import time

from app_tree import UnfinishedLeaf
from mcts import MCTNode, mct_search
from nmcs import nested_mc_search
from test_montecarlo import make_env_stack, make_env
from tree_node import ChooseKTNode, StackNode
from tree_stats import TreeStats
from utils import experiment_eval

if __name__ == "__main__":
    #make_env = make_env
    make_env = make_env_stack

    if False:
        # Nested MC Search
        def one_iteration(env):
            evals_before = env.count_evals()
            time_before = time.time()
            #root = ChooseKTNode(UnfinishedLeaf(), 5)
            root = StackNode([])
            indiv = nested_mc_search(root,
                                     max_level=1,
                                     fitness=env.fitness,
                                     finish=env.finish,
                                     is_finished=env.is_finished,
                                     successors=env.successors,
                                     advance=env.advance)
            return env.fitness(indiv), env.count_evals() - evals_before, time.time() - time_before


        experiment_eval(one_iteration, repeat=1, processes=1, make_env=make_env)

    if False:
        # MCTS
        def one_iteration(env):
            evals_before = env.count_evals()
            time_before = time.time()
            #root = MCTNode(ChooseKTNode(UnfinishedLeaf(), 5))
            root = MCTNode(StackNode([]))
            mct_search(root, expand_visits=8, num_steps=100,
                       fitness=env.fitness,
                       finish=env.finish,
                       is_finished=env.is_finished,
                       successors=env.successors)
            return root.best_score, env.count_evals() - evals_before, time.time() - time_before


        experiment_eval(one_iteration, repeat=1, processes=1, make_env=make_env)

    if not False:
        # MCTS - one run
        env = make_env()
        # tracer_deco.enable_tracer = True
        # random.seed(5)

        #root = MCTNode(ChooseKTNode(UnfinishedLeaf(), 5))
        root = MCTNode(StackNode([]))
        tree_stats = TreeStats()
        mct_search(root, expand_visits=2, num_steps=100,
                   fitness=env.fitness,
                   finish=env.finish,
                   is_finished=env.is_finished,
                   successors=env.successors,
                   tree_stats=tree_stats)
        print('=' * 20)
        print(root.pretty_str())
        print(tree_stats.pretty_str())

#!/usr/bin/env pypy3
import argparse
import time

from app_tree import UnfinishedLeaf
from domain_koza_apptree import make_env_app_tree
from domain_koza_stack import make_env_stack
from mcts import MCTNode, mct_search
from nmcs import nested_mc_search
from tree_node import ChooseKTNode, StackNode
from utils import experiment_eval


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--proc', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--k', type=int, default=5)

    parser.add_argument('--mcts', action='store_true', default=False)
    parser.add_argument('--nmcs', action='store_true', default=False)

    parser.add_argument('--stack', action='store_true', default=False)
    parser.add_argument('--app-tree', action='store_true', default=False)

    parser.add_argument('--nmcs-level', type=int, default=1)
    parser.add_argument('--mcts-expand', type=int, default=8)
    parser.add_argument('--mcts-num-steps', type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    assert not (args.mcts and args.nmcs)
    assert args.mcts or args.nmcs
    assert not (args.stack and args.app_tree)
    assert args.stack or args.app_tree

    if args.app_tree:
        make_env = make_env_app_tree
    elif args.stack:
        make_env = lambda: make_env_stack(args.k)
    else:
        assert False

    if args.nmcs:
        # Nested MC Search
        def one_iteration(worker_env):
            env = make_env()
            evals_before = env.count_evals()
            assert not evals_before
            time_before = time.time()
            if args.app_tree:
                root = ChooseKTNode(UnfinishedLeaf(), args.k)
            elif args.stack:
                root = StackNode([])
            else:
                assert False

            indiv = nested_mc_search(root,
                                     max_level=args.nmcs_level,
                                     fitness=env.fitness,
                                     finish=env.finish,
                                     is_finished=env.is_finished,
                                     successors=env.successors,
                                     advance=env.advance,
                                     early_end_test=env.early_end_test)

            env.cache.print_self("AT END")
            return env.fitness(indiv), env.count_evals() - evals_before, time.time() - time_before


        experiment_eval(one_iteration, repeat=args.repeat, processes=args.proc, make_env=lambda: None)

    if args.mcts:
        # MCTS
        def one_iteration(worker_env):
            env = make_env()
            evals_before = env.count_evals()
            time_before = time.time()
            if args.app_tree:
                root = MCTNode(ChooseKTNode(UnfinishedLeaf(), args.k))
            elif args.stack:
                root = MCTNode(StackNode([]))
            else:
                assert False

            mct_search(root, expand_visits=args.mcts_expand, num_steps=args.mcts_num_steps,
                       fitness=env.fitness,
                       finish=env.finish,
                       is_finished=env.is_finished,
                       successors=env.successors,
                       early_end_test=env.early_end_test
                       )

            env.cache.print_self("AT END")
            return root.best_score, env.count_evals() - evals_before, time.time() - time_before


        experiment_eval(one_iteration, repeat=args.repeat, processes=args.proc, make_env=lambda: None)

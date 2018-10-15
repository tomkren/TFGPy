#!/usr/bin/env pypy3
import argparse
import time

import domain_koza_apptree
import domain_koza_stack
import domain_parity_apptree
import domain_parity_stack
import domain_physics
import domain_physics_smart
import domain_primes_apptree
import stack
from app_tree import UnfinishedLeaf
from mcts import MCTNode, mct_search, C_UCT_EXPLORE_DEFAULT
from nmcs import nested_mc_search
from tree_node import ChooseKTNode, StackNode, UFTNode, MaxKTNode
from utils import experiment_eval

D_PARITY = 'parity'

D_PRIMES = 'primes'

D_KOZA_POLY = 'koza_poly'

D_PHYSICS = 'physics'

D_PHYSICS_SMART = 'physics_smart'

APP_T_MAX_K = 'max_k'

APP_T_FIXED_K = 'fix_k'

APP_T_CHOOSE_K = 'choose_k'

import random

random.seed(3)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--proc', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--k', type=int, default=5)

    parser.add_argument('--smart-uf', '-s', action='store_true', default=False)

    parser.add_argument('--mcts', action='store_true', default=False)
    parser.add_argument('--nmcs', action='store_true', default=False)

    parser.add_argument('--domain', type=str, choices=[D_KOZA_POLY, D_PRIMES, D_PARITY, D_PHYSICS_SMART, D_PHYSICS], default=D_KOZA_POLY)
    parser.add_argument('--stack', action='store_true', default=False)
    parser.add_argument('--app-tree', type=str, choices=['', APP_T_CHOOSE_K, APP_T_FIXED_K, APP_T_MAX_K],
                        default='')

    parser.add_argument('--nmcs-level', type=int, default=1)

    parser.add_argument('--mcts-expand', type=int, default=8)
    parser.add_argument('--mcts-num-steps', type=int, default=100)
    parser.add_argument('--mcts-sample-by-urgency', action='store_true', default=False)
    parser.add_argument('--mcts-urgency-method', choices=['best', 'avg', '(avg+best)/2'], default='best')
    parser.add_argument('--mcts-urgency-c-uct', type=float, default=C_UCT_EXPLORE_DEFAULT)

    parser.add_argument('--print-size-hist', action='store_true', default=False)
    return parser.parse_args()


def print_k_histogram(domain):
    #print()
    #print(len(stack.missings))
    #print(sum(stack.missings)/len(stack.missings))
    print("k-size histogram, sum=", sum(domain.size_d.values()))
    print('\n'.join("%s = %s" % (a, b) for a, b in sorted(domain.size_d.items())))


def construct_root_node(args, env):
    if args.app_tree == APP_T_CHOOSE_K:
        root = ChooseKTNode(uf_factory(env), args.k)
    elif args.app_tree == APP_T_FIXED_K:
        root = UFTNode(uf_factory(env), args.k)
    elif args.app_tree == APP_T_MAX_K:
        root = MaxKTNode(uf_factory(env), args.k)
    elif args.stack:
        root = StackNode([])
    else:
        assert False
    return root

if __name__ == "__main__":
    args = parse_args()
    print(args)

    assert not (args.mcts and args.nmcs)
    assert args.mcts or args.nmcs
    assert not (args.stack and args.app_tree)

    if args.app_tree:
        if args.domain == D_KOZA_POLY:
            domain = domain_koza_apptree
        elif args.domain == D_PARITY:
            domain = domain_parity_apptree
        elif args.domain == D_PRIMES:
            domain = domain_primes_apptree
        elif args.domain == D_PHYSICS:
            domain = domain_physics
        elif args.domain == D_PHYSICS_SMART:
            domain = domain_physics_smart
        else:
            assert False
        make_env = lambda: domain.make_env_app_tree(smart=args.smart_uf)
    elif args.stack:
        if args.domain == D_KOZA_POLY:
            domain = domain_koza_stack
        elif args.domain == D_PARITY:
            domain = domain_parity_stack
        else:
            assert False
        make_env = lambda: domain.make_env_stack(args.k)
    else:
        assert False

    uf_factory = lambda env: UnfinishedLeaf()
    if args.smart_uf:
        uf_factory = lambda env: UnfinishedLeaf(typ=env.goal)

    if args.nmcs:
        # Nested MC Search
        def one_iteration(worker_env):
            env = make_env()
            evals_before = env.count_evals()
            assert not evals_before
            time_before = time.time()
            root = construct_root_node(args, env)
            indiv = nested_mc_search(root,
                                     max_level=args.nmcs_level,
                                     fitness=env.fitness,
                                     finish=env.finish,
                                     is_finished=env.is_finished,
                                     successors=env.successors,
                                     advance=env.advance,
                                     early_end_test=env.early_end_test)

            env.cache.print_self("AT END")
            if args.print_size_hist:
                print_k_histogram(domain)
            return env.fitness(indiv), env.count_evals() - evals_before, time.time() - time_before


        experiment_eval(one_iteration, repeat=args.repeat, processes=args.proc, make_env=lambda: None)

    if args.mcts:
        # MCTS
        def one_iteration(worker_env):
            env = make_env()
            evals_before = env.count_evals()
            time_before = time.time()
            root = construct_root_node(args, env)
            mct_root = MCTNode(root)
            mct_search(mct_root, expand_visits=args.mcts_expand, num_steps=args.mcts_num_steps,
                       fitness=env.fitness,
                       finish=env.finish,
                       is_finished=env.is_finished,
                       successors=env.successors,
                       early_end_test=env.early_end_test,
                       sample_by_urgency=args.mcts_sample_by_urgency,
                       urgency_method=args.mcts_urgency_method,
                       urgency_c_uct_explore=args.mcts_urgency_c_uct)

            env.cache.print_self("AT END")
            if args.print_size_hist:
                print_k_histogram(domain)
            return mct_root.best_score, env.count_evals() - evals_before, time.time() - time_before


        experiment_eval(one_iteration, repeat=args.repeat, processes=args.proc, make_env=lambda: None)

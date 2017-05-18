import random
import time

import generator
import test_montecarlo
import utils
from app_tree import UnfinishedLeaf
from mcts import MCTNode, mct_search
from nmcs import dfs_advance_skeleton, nested_mc_search
from tree_node import UFTNode, ChooseKTNode
from utils import experiment_eval


def make_env():
    goal, gamma, fitness, count_evals = test_montecarlo.regression_domain_koza_poly()

    class Environment:
        def __init__(self, goal, gamma, fitness, count_evals, gen):
            self.goal = goal
            self.gamma = gamma
            self.fitness = fitness
            self.count_evals = count_evals
            self.gen = gen

            #
            #  now define tree-searching functions in this env
            #

            @utils.pp_function('fitness()')
            def ffitness(tree):
                assert isinstance(tree, UFTNode)
                # make sure we only run fitness on finished,
                # fully typed trees
                assert tree.uf_tree.typ is not None
                return self.fitness(tree.uf_tree)

            @utils.pp_function('finish()')
            def ffinish(tree):
                assert isinstance(tree, UFTNode)
                assert tree.uf_tree.typ is None

                finished_tree = self.gen.gen_one_uf(tree.uf_tree, tree.k, self.goal)
                assert finished_tree.typ is not None
                return UFTNode(finished_tree, tree.k)

            @utils.pp_function('successors()')
            def fsuccessors(tree):
                if isinstance(tree, UFTNode):
                    return [UFTNode(c, tree.k) for c in tree.uf_tree.successors(self.gen, tree.k, self.goal)]
                if isinstance(tree, ChooseKTNode):
                    return [UFTNode(tree.uf_tree, k) for k in range(2, tree.max_k + 1)]
                assert False

            @utils.pp_function('advance()')
            def fadvance(tree, finished_tree):
                assert isinstance(tree, UFTNode) or isinstance(tree, ChooseKTNode)
                assert isinstance(finished_tree, UFTNode)
                new_uf_tree = dfs_advance_skeleton(tree.uf_tree, finished_tree.uf_tree)
                if new_uf_tree is None:
                    return None
                return UFTNode(new_uf_tree, finished_tree.k)

            self.t_fitness = ffitness
            self.t_finish = ffinish
            self.t_successors = fsuccessors
            self.t_advance = fadvance

    return Environment(goal, gamma, fitness, count_evals, generator.Generator(gamma))


if __name__ == "__main__":
    if False:
        env = make_env()
        random.seed(5)
        indiv = env.gen.gen_one(30, env.goal)
        print(indiv.eval_str())
        print(env.fitness(indiv))

    if not False:
        # Nested MC Search
        def one_iteration(env):
            evals_before = env.count_evals()
            time_before = time.time()
            indiv = nested_mc_search(ChooseKTNode(UnfinishedLeaf(), 20),
                                     max_level=3,
                                     fitness=env.t_fitness,
                                     finish=env.t_finish,
                                     successors=env.t_successors,
                                     advance=env.t_advance)
            return env.fitness(indiv.uf_tree), env.count_evals() - evals_before, time.time() - time_before


        experiment_eval(one_iteration, repeat=2, processes=2, make_env=make_env)

    if False:
        # MCTS
        def one_iteration(env):
            evals_before = env.count_evals()
            time_before = time.time()
            root = MCTNode(ChooseKTNode(UnfinishedLeaf(), 20))
            mct_search(root, expand_visits=8, num_steps=10000,
                       fitness=env.t_fitness,
                       finish=env.t_finish,
                       successors=env.t_successors)
            return root.best_score, env.count_evals() - evals_before, time.time() - time_before


        experiment_eval(one_iteration, repeat=10, processes=2, make_env=make_env)

    if False:
        # MCTS - one run
        env = make_env()
        # tracer_deco.enable_tracer = True
        # TODO DEBUG
        # very slow for large k = 20
        # random.seed(5)
        root = MCTNode(ChooseKTNode(UnfinishedLeaf(), 20))
        tstats = mct_search(root, expand_visits=2, num_steps=20000,
                            fitness=env.t_fitness,
                            finish=env.t_finish,
                            successors=env.t_successors)
        print('=' * 20)
        print(root.pretty_str())
        print(tstats.pretty_str())

import math
import random
from functools import wraps

import sys

import generator
import test_montecarlo
import utils
from app_tree import App, UnfinishedLeaf, UNFINISHED_APP
from app_tree import Leaf
import tracer_deco
from utils import experiment_eval, PPFunction

C_UCT_EXPLORE = 0.5


def dfs_advance_skeleton(old_skeleton, finished_tree):
    """Advance `old_skeleton` one step towards the `finished_tree`."""
    assert finished_tree.is_finished()
    assert old_skeleton.is_skeleton_of(finished_tree)

    if isinstance(old_skeleton, App):
        assert isinstance(finished_tree, App)
        maybe_new_fun = dfs_advance_skeleton(old_skeleton.fun, finished_tree.fun)
        if maybe_new_fun is not None:
            return App(maybe_new_fun, old_skeleton.arg)

        maybe_new_arg = dfs_advance_skeleton(old_skeleton.arg, finished_tree.arg)
        if maybe_new_arg is not None:
            return App(old_skeleton.fun, maybe_new_arg)
        return None

    if isinstance(old_skeleton, UnfinishedLeaf):
        if isinstance(finished_tree, Leaf):
            return finished_tree
        if isinstance(finished_tree, App):
            return UNFINISHED_APP
        assert False

    if isinstance(old_skeleton, Leaf):
        return None

    assert False


#
#   Nested Monte Carlo Search
#
#   adapted from:
#       Cazenave, Tristan: Monte-carlo expression discovery.
#       International Journal on Artificial Intelligence Tools 22.01 (2013): 1250035. APA


def nested_mc_search(gen, goal_typ, k, max_level, fitness, advance_skeleton=None):
    # @tracer_deco(print_from_arg=0)
    def nested_mc_search_raw(level, k, uf_tree):
        # print(k, level, uf_tree, flush=True)

        # if the input tree is already finished, evaluate it
        if uf_tree.is_finished():
            uf_tree.score = fitness(uf_tree)
            return uf_tree

        # on level 0, finish the tree uniformly randomly with size k and evaluate
        if level == 0:
            tree = gen.gen_one_uf(uf_tree, k, goal_typ)
            tree.score = fitness(tree)
            return tree

        # best finished tree found
        best_tree = None
        while uf_tree is not None:
            for uf_successor in uf_tree.successors(gen, k, goal_typ):
                # print("SUCC:", uf_successor)
                tree = nested_mc_search_raw(level - 1, k, uf_successor)
                if best_tree is None or tree.score > best_tree.score:
                    best_tree = tree
                    # print("BEST NEMC", best.score)
            # the successors must be nonempty, so:
            assert best_tree is not None

            uf_tree_new = advance_skeleton(uf_tree, best_tree)
            # advance_skeleton should return None
            # when the uf_tree (old skeleton) is already finished
            if uf_tree_new is not None:
                assert uf_tree_new.is_skeleton_of(best_tree)
                # the advance_skeleton should concretize just one symbol
                assert uf_tree.count_finished_nodes() + 1 == uf_tree_new.count_finished_nodes()
            uf_tree = uf_tree_new

        return best_tree

    if advance_skeleton is None:
        advance_skeleton = dfs_advance_skeleton

    return nested_mc_search_raw(max_level, k, UnfinishedLeaf())


#
#   MCTS - Monte Carlo Tree Search
#

class MCTNode:
    def __init__(self, uf_tree):
        self.uf_tree = uf_tree
        self.children = None
        self.visits = 0

        self.best = None
        self.best_score = 0.0

    def __str__(self):
        return "MCTNode<%s>" % (self.uf_tree)

    def update_best(self, tree, score):
        if self.best_score < score:
            self.best = tree
            self.best_score = score

    def urgency(self, total_visits):
        # TODO evaluate
        return (1 - C_UCT_EXPLORE) * self.best_score + C_UCT_EXPLORE * math.sqrt(
            math.log(total_visits) / (1 + self.visits))

    @tracer_deco.tracer_deco(log_ret=True, ret_pp=lambda l: ", ".join(map(str, l)))
    def expand(self, expander):
        if not self.uf_tree.is_finished():
            self.children = [MCTNode(child_tree) for child_tree in expander(self.uf_tree)]
        return self.children


@tracer_deco.tracer_deco()
def mct_descend(node, expand_visits, expander, sample_by_urgency=False):
    node.visits += 1
    nodes = [node]

    expanded = False
    while nodes[-1].children is not None:
        if not sample_by_urgency:
            # Pick the most urgent child
            children = list(nodes[-1].children)
            # symmetry breaking for children with the same urgency
            random.shuffle(children)
            node = max(children, key=lambda child_node: child_node.urgency(node.visits))
        else:
            # JM: this is probably worse
            # tested on Koza's polynomial domain with
            # 200 repetitions of MCTS with 1000 playouts
            children = nodes[-1].children
            urgencies = [child_node.urgency(node.visits) for child_node in children]
            node = utils.sample_by_scores(children, urgencies)

        nodes.append(node)
        node.visits += 1
        if node.children is None and node.visits >= expand_visits and not expanded:
            expanded = True
            node.expand(expander)

    return nodes


@tracer_deco.tracer_deco(log_ret=True, ret_pp=(lambda t: "%.3f %s" % (t[1], t[0])))
def mct_playout(node, gen_one_uf, fitness):
    if node.uf_tree.is_finished():
        tree = node.uf_tree
    else:
        tree = gen_one_uf(node.uf_tree)

    return tree, fitness(tree)


def mct_update(nodes, tree, score):
    for node in reversed(nodes):
        node.update_best(tree, score)


@tracer_deco.tracer_deco(force_enable=True)
def mct_search(node, gen, k, goal_typ, expand_visits, num_steps, fitness):
    gen_one_uf = PPFunction(lambda uf_tree: gen.gen_one_uf(uf_tree, k, goal_typ),
                            pp_name='gen_one_uf()')
    expander = PPFunction(lambda uf_tree: uf_tree.successors(gen, k, goal_typ),
                          pp_name='gen_one_uf()')

    if node.children is None:
        node.expand(expander)

    i = 0
    while i < num_steps:
        i += 1
        nodes = mct_descend(node, expand_visits, expander)
        tree, score = mct_playout(nodes[-1], gen_one_uf, fitness)
        mct_update(nodes, tree, score)


if __name__ == "__main__":
    def make_env():
        goal, gamma, fitness, count_evals = test_montecarlo.regression_domain_koza_poly()

        class Environment:
            def __init__(self, goal, gamma, fitness, count_evals, gen):
                self.goal = goal
                self.gamma = gamma
                self.fitness = fitness
                self.count_evals = count_evals
                self.gen = gen

        return Environment(goal, gamma, PPFunction(fitness, pp_name='fitness()'),
                           count_evals, generator.Generator(gamma))


    if False:
        env = make_env()
        random.seed(5)
        indiv = env.gen.gen_one(30, env.goal)
        print(indiv.eval_str())
        print(env.fitness(indiv))

    if False:
        def one_iteration(env):
            evals_before = env.count_evals()
            indiv = nested_mc_search(env.gen, env.goal, k=10, max_level=2, fitness=env.fitness)
            return env.fitness(indiv), env.count_evals() - evals_before


        experiment_eval(one_iteration, repeat=20, processes=3, make_env=make_env)

    if False:
        def one_iteration(env):
            evals_before = env.count_evals()
            root = MCTNode(UnfinishedLeaf())
            mct_search(root, env.gen, k=10, goal_typ=env.goal, expand_visits=8, num_steps=1000, fitness=env.fitness)
            return root.best_score, env.count_evals() - evals_before


        experiment_eval(one_iteration, repeat=200, processes=2, make_env=make_env)

    if True:
        env = make_env()
        tracer_deco.enable_tracer = True
        # TODO DEBUG
        # very slow for large k = 20
        random.seed(5)
        root = MCTNode(UnfinishedLeaf())
        mct_search(root, env.gen, k=10, goal_typ=env.goal, expand_visits=8, num_steps=50, fitness=env.fitness)
        print('=' * 20)
        print(root.best)
        print(root.best_score)
        print(root.visits, root.uf_tree)
        print("children")
        for c in root.children[0].children:
            print(c.visits, "%.3f" % c.urgency(root.visits), c.uf_tree)




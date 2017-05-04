import math
import random

import generator
import test_montecarlo
from app_tree import App, UnfinishedLeaf, UNFINISHED_APP
from app_tree import Leaf
from utils import experiment_eval

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


def nested_mc_search(gen, goal_typ, max_k, max_level, fitness, advance_skeleton=None):
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

    best = None
    for k in range(1, max_k + 1):
        print("=" * 10, "k=", k)
        this = nested_mc_search_raw(max_level, k, UnfinishedLeaf())
        # print(this)
        if best is None or this.score > best.score:
            best = this
            print("BEST K", best.score)

    return best


#
#   MCTS - Monte Carlo Tree Search
#

class MCTNode:
    def __init__(self, uf_tree):
        self.uf_tree = uf_tree
        self.children = None
        self.visits = 1

        self.best = None
        self.best_score = 0.0

    def update_best(self, tree, score):
        if self.best_score < score:
            self.best = tree
            self.best_score = score

    def urgency(self, total_visits):
        # TODO evaluate
        assert self.visits >= 1
        return (1 - C_UCT_EXPLORE) * self.best_score + C_UCT_EXPLORE * math.sqrt(math.log(total_visits) / self.visits)

    def expand(self, expander):
        if not self.uf_tree.is_finished():
            self.children = [MCTNode(child_tree) for child_tree in expander(self.uf_tree)]


def mct_descend(node, expander, expand_visits):
    node.visits += 1
    nodes = [node]

    while nodes[-1].children is not None:
        # Pick the most urgent child
        children = list(nodes[-1].children)
        # symmetry breaking for children with the same urgency
        random.shuffle(children)
        node = max(children, key=lambda child_node: child_node.urgency(node.visits))
        nodes.append(node)

        node.visits += 1
        if node.children is None and node.visits >= expand_visits:
            node.expand(expander)

    return nodes


def mct_playout(node, gen_one_uf):
    if node.uf_tree.is_finished():
        tree = node.uf_tree
    else:
        tree = gen_one_uf(node.uf_tree)

    return tree, fitness(tree)


def mct_update(nodes, tree, score):
    for node in reversed(nodes):
        node.update_best(tree, score)


def mct_search(node, gen, k, goal_typ, expand_visits, num_steps):
    gen_one_uf = lambda uf_tree: gen.gen_one_uf(uf_tree, k, goal_typ)
    expander = lambda uf_tree: uf_tree.successors(gen, k, goal_typ)

    if node.children is None:
        node.expand(expander)

    i = 0
    while i < num_steps:
        i += 1
        nodes = mct_descend(node, expander, expand_visits)
        tree, score = mct_playout(nodes[-1], gen_one_uf)
        mct_update(nodes, tree, score)


if __name__ == "__main__":
    goal, gamma, fitness, count_evals = test_montecarlo.regression_domain_koza_poly()

    gen = generator.Generator(gamma)
    if False:
        random.seed(5)
        indiv = gen.gen_one(30, goal)
        print(indiv.eval_str())
        print(fitness(indiv))

    if False:
        values = []
        for i in range(1):
            print("=" * 10, i, "=" * 10)
            indiv = nested_mc_search(gen, goal, max_k=5, max_level=3, fitness=fitness)
            print(indiv.eval_str())
            fi = fitness(indiv)
            values.append(fi)
            print(fi)

        print()
        print(sum(values) / len(values), min(values), max(values))
        print(count_evals())


    # TODO DEBUG
    # very slow for large k = 20
    for expands in range(2, 11):
        def one_iteration():
            root = MCTNode(UnfinishedLeaf())
            mct_search(root, gen, k=10, goal_typ=goal, expand_visits=expands, num_steps=1000)
            return root.best_score


        print('=' * 10)
        print('expands=%d' % expands)
        print('=' * 10)
        experiment_eval(one_iteration, 20)

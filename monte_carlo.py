import math
import random
from functools import wraps
from itertools import chain

import time

import generator
import test_montecarlo
import tracer_deco
import utils
from app_tree import App, UnfinishedLeaf, UNFINISHED_APP, AppTree
from app_tree import Leaf
from utils import experiment_eval

# 0.5 is a reasonable value
# but it depends on the fitness function at hand
C_UCT_EXPLORE = 0.5


# advance_skeleton should return None
# when the uf_tree (old skeleton) is already finished

def check_skeleton_advancer(f):
    @wraps(f)
    def g(old_skeleton, finished_tree):
        assert finished_tree.is_finished()
        assert old_skeleton.is_skeleton_of(finished_tree)
        new_skeleton = f(old_skeleton, finished_tree)
        if new_skeleton is not None:
            assert new_skeleton.is_skeleton_of(finished_tree)
            # the advance_skeleton should concretize just one symbol
            assert old_skeleton.count_finished_nodes() + 1 == new_skeleton.count_finished_nodes()

        return new_skeleton

    return g


@check_skeleton_advancer
def dfs_advance_skeleton(old_skeleton, finished_tree):
    """Advance `old_skeleton` one step towards the `finished_tree`."""

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


class TNode:
    def __str__(self):
        return "TNode<>"

    def is_finished(self):
        raise NotImplementedError


class UFTNode(TNode):
    def __init__(self, uf_tree, k):
        self.uf_tree = uf_tree
        self.k = k

    def __str__(self):
        return "UFTNode<k=%d, %s>" % (self.k, self.uf_tree)

    def is_finished(self):
        return self.uf_tree.is_finished()


class ChooseKTNode(TNode):
    def __init__(self, uf_tree, max_k):
        self.uf_tree = uf_tree
        self.max_k = max_k

    def __str__(self):
        return "ChooseKTNode<max_k=%d, %s>" % (self.max_k, self.uf_tree)

    def is_finished(self):
        return self.uf_tree.is_finished()


#
#   Nested Monte Carlo Search
#
#   adapted from:
#       Cazenave, Tristan: Monte-carlo expression discovery.
#       International Journal on Artificial Intelligence Tools 22.01 (2013): 1250035. APA


@tracer_deco.tracer_deco(force_enable=True)
def nested_mc_search(root_tree, max_level, fitness, finish, successors, advance):
    # @tracer_deco(print_from_arg=0)
    def nested_mc_search_raw(level, tree):
        # if the input tree is already finished, evaluate it
        if tree.is_finished():
            tree.score = fitness(tree)
            return tree

        if level == 0:
            finished_tree = finish(tree)
            finished_tree.score = fitness(finished_tree)
            return finished_tree

        # best finished tree found
        best_finished_tree = None
        while tree is not None:
            assert tree is not None
            for tree_successor in successors(tree):
                finished_tree = nested_mc_search_raw(level - 1, tree_successor)
                if best_finished_tree is None or finished_tree.score > best_finished_tree.score:
                    best_finished_tree = finished_tree

            # the successors must be nonempty, so:
            assert best_finished_tree is not None
            tree = advance(tree, best_finished_tree)

        return best_finished_tree

    return nested_mc_search_raw(max_level, root_tree)


#
#   Statistics on subtrees
#       - to be used for heuristics during search / smart playouts
#


class RunningStat:
    def __init__(self, count=0, sum=0):
        self.count = count
        self.sum = sum
        self.biggest = None
        self.smallest = None

    def add(self, value):
        self.count += 1
        self.sum += value
        if self.biggest is None or value > self.biggest:
            self.biggest = value
        if self.smallest is None or value < self.smallest:
            self.smallest = value

    def avg(self):
        if not self.count:
            raise ValueError
        return self.sum / self.count


class Stats:
    def __init__(self):
        self.total = RunningStat()
        self.by_tree = {}

    def update(self, tree, score):
        self.total.add(score)
        self.by_tree.setdefault(tree, RunningStat()).add(score)


class TreeStats:
    def __init__(self):
        self.typ2size2stats = {}

    def update(self, root, score):
        assert isinstance(root, AppTree)

        def update_one(tree):
            assert tree.typ is not None
            assert tree.is_finished()
            counts = tree.count_nodes()
            k = counts[Leaf]

            size2stats = self.typ2size2stats.setdefault(tree.typ, {})
            size2stats.setdefault(k, Stats()).update(tree, score)

        root.map_reduce(update_one, (lambda *args: None))

    def pretty_str(self):
        l = []
        for typ, size2stats in self.typ2size2stats.items():
            l.append('=' * 10 + str(typ) + '=' * 10)
            for k, stats in sorted(size2stats.items()):
                t, rs = max(stats.by_tree.items(), key=(
                    lambda t: t[1].avg()
                    # lambda t: t[1].biggest
                ))
                l.append("k=%d %d %.3f %s" % (k, rs.count, rs.avg(), t))
        return '\n'.join(l)


#
#   MCTS - Monte Carlo Tree Search
#

class MCTNode:
    def __init__(self, tree):
        self.tree = tree

        # flag for marking that
        # the subtree is fully expanded (self.uf_tree finished)
        self.finished_flag = False
        self.children = None
        self.visits = 0

        self.best = None
        self.score_sum = 0
        self.score_num = 0
        self.best_score = 0

    def __str__(self):
        return "MCTNode(%s)" % (self.tree)

    def update_best(self, new, score):
        self.score_sum += score
        self.score_num += 1
        assert self.score_num == self.visits
        if self.best_score < score:
            self.best = new
            self.best_score = score

        if (self.tree.is_finished()
            or (self.children is not None
                and all(c.finished_flag for c in self.children))):
            # this prevents further searching through this (now finished) node
            self.finished_flag = True

    def urgency(self, total_visits, method='best'):
        if self.finished_flag:
            return 0.0

        exploration = C_UCT_EXPLORE * math.sqrt(math.log(total_visits) / (1 + self.visits))

        # from preliminary tests, best is the best
        if method == 'best':
            exploatation = (1 - C_UCT_EXPLORE) * self.best_score
        elif method == 'avg':
            exploatation = (1 - C_UCT_EXPLORE) * self.score_sum / (1 + self.score_num)
        elif method == '(avg+best)/2':
            exploatation = (1 - C_UCT_EXPLORE) * 0.5 * (self.best_score + self.score_sum / (1 + self.score_num))
        else:
            assert False

        return exploatation + exploration

    # @tracer_deco.tracer_deco(log_ret=True, ret_pp=lambda l: ", ".join(map(str, l)))
    def expand(self, successors):
        if not self.tree.is_finished():
            self.children = [MCTNode(child_tree) for child_tree in successors(self.tree)]

        return self.children

    def pretty_str(self):
        l = []
        l.append("best=%s" % (self.best))
        l.append("%d %.3f  %s" % (self.visits, self.best_score, self.tree))
        # pad the visits, s.t. they have the same len
        fs = "%%%dd" % (len(str(self.visits)))
        for c in self.children:
            l.append("%s %.3f = %.3f  %s" % (fs % (c.visits), c.best_score, c.urgency(self.visits), c.tree))
        return '\n'.join(l)


@tracer_deco.tracer_deco()
def mct_descend(node, expand_visits, successors, sample_by_urgency=False):
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
            node.expand(successors)

    return nodes


@tracer_deco.tracer_deco(log_ret=True, ret_pp=(lambda t: "%.3f %s" % (t[1], t[0])))
def mct_playout(node, finish, fitness):
    assert not node.finished_flag
    finished_tree = finish(node.tree)
    return finished_tree, fitness(finished_tree)


def mct_update(nodes, tree, score):
    # the reverse order is necessary, since we
    # need the children to be updated before we update
    # parent (e.g. for correct node.is_finished update)
    for node in reversed(nodes):
        node.update_best(tree, score)


@tracer_deco.tracer_deco(force_enable=True)
def mct_search(node, num_steps, fitness, finish, successors, expand_visits=8, tree_stats=None):
    if node.children is None:
        node.expand(successors)

    i = 0
    while i < num_steps and not node.finished_flag:
        nodes = mct_descend(node, expand_visits, successors)
        # no need to do playout if the subtree is
        # fully expanded and
        if nodes[-1].finished_flag:
            continue
        tree, score = mct_playout(nodes[-1], finish, fitness)
        mct_update(nodes, tree, score)
        if tree_stats is not None:
            tree_stats.update(tree.uf_tree, score)
        i += 1

    return tree_stats


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


    if False:
        env = make_env()
        random.seed(5)
        indiv = env.gen.gen_one(30, env.goal)
        print(indiv.eval_str())
        print(env.fitness(indiv))

    if False:
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

    if not False:
        # MCTS
        def one_iteration(env):
            evals_before = env.count_evals()
            time_before = time.time()
            root = MCTNode(ChooseKTNode(UnfinishedLeaf(), 20))
            tree_stats = None
            #env.gen.tree_stats = tree_stats
            mct_search(root, expand_visits=8, num_steps=10000,
                       fitness=env.t_fitness,
                       finish=env.t_finish,
                       successors=env.t_successors,
                       tree_stats=tree_stats)
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
        tree_stats = TreeStats()
        env.gen.tree_stats = tree_stats
        mct_search(root, expand_visits=2, num_steps=1000,
                   fitness=env.t_fitness,
                   finish=env.t_finish,
                   successors=env.t_successors,
                   tree_stats=tree_stats)
        print('=' * 20)
        print(tree_stats.pretty_str())
        print('=' * 20)
        print(root.pretty_str())

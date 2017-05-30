import math
import random

import tracer_deco
import utils

# 0.5 is a reasonable value
# but it depends on the fitness function at hand
C_UCT_EXPLORE = 0.5


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

    def update_best(self, new, score, is_finished):
        self.score_sum += score
        self.score_num += 1
        assert self.score_num == self.visits
        if self.best_score < score:
            self.best = new
            self.best_score = score

        if (is_finished(self.tree)
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
    def expand(self, successors, is_finished):
        if not is_finished(self.tree):
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
def mct_descend(node, expand_visits, successors, is_finished, sample_by_urgency=False):
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
            node.expand(successors, is_finished)

    return nodes


@tracer_deco.tracer_deco(log_ret=True, ret_pp=(lambda t: "%.3f %s" % (t[1], t[0])))
def mct_playout(node, finish, fitness):
    assert not node.finished_flag
    finished_tree = finish(node.tree)
    return finished_tree, fitness(finished_tree)


def mct_update(nodes, tree, score, is_finished):
    # the reverse order is necessary, since we
    # need the children to be updated before we update
    # parent (e.g. for correct node.is_finished update)
    for node in reversed(nodes):
        node.update_best(tree, score, is_finished)


@tracer_deco.tracer_deco(force_enable=True)
def mct_search(node, num_steps, fitness, finish, is_finished, successors, expand_visits=8, tree_stats=None):
    if node.children is None:
        node.expand(successors, is_finished)

    i = 0
    while i < num_steps and not node.finished_flag:
        nodes = mct_descend(node, expand_visits, successors, is_finished)
        # no need to do playout if the subtree is
        # fully expanded and
        if nodes[-1].finished_flag:
            continue
        tree, score = mct_playout(nodes[-1], finish, fitness)
        mct_update(nodes, tree, score, is_finished)
        if tree_stats is not None:
            tree_stats.update(tree, score)
        i += 1

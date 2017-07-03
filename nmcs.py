from functools import wraps

import tracer_deco
from app_tree import App, UnfinishedLeaf, Leaf, UNFINISHED_APP


#
#   Nested Monte Carlo Search
#
#   adapted from:
#       Cazenave, Tristan: Monte-carlo expression discovery.
#       International Journal on Artificial Intelligence Tools 22.01 (2013): 1250035. APA


@tracer_deco.tracer_deco(force_enable=True)
def nested_mc_search(root_tree, max_level, fitness, finish, is_finished, successors, advance,
                     early_end_test=lambda s: False):
    # @tracer_deco(print_from_arg=0)
    def nested_mc_search_raw(level, tree):
        # if the input tree is already finished, evaluate it
        if level == 0 or is_finished(tree):
            tree = finish(tree)
            tree.score = fitness(tree)
            return tree

        # best finished tree found
        best_finished_tree = None
        while tree is not None:
            assert tree is not None
            for tree_successor in successors(tree):
                finished_tree = nested_mc_search_raw(level - 1, tree_successor)
                if best_finished_tree is None or finished_tree.score > best_finished_tree.score:
                    best_finished_tree = finished_tree
                if early_end_test(best_finished_tree.score):
                    return best_finished_tree

            # the successors must be nonempty, so:
            assert best_finished_tree is not None
            assert is_finished(best_finished_tree)
            tree = advance(tree, best_finished_tree)

        return best_finished_tree

    return nested_mc_search_raw(max_level, root_tree)


# advance_skeleton should return None
# when the uf_tree (old skeleton) is already finished

def check_skeleton_advancer(f):
    @wraps(f)
    def g(old_skeleton, finished_tree):
        assert old_skeleton.is_skeleton_of(finished_tree)
        new_skeleton = f(old_skeleton, finished_tree)
        if new_skeleton is not None:
            assert new_skeleton.is_skeleton_of(finished_tree)
            # the advance_skeleton should concretize just one symbol
            old = old_skeleton.count_finished_nodes()
            new = new_skeleton.count_finished_nodes()
            assert old + 1 == new

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

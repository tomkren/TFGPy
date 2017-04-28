from app_tree import App, UnfinishedLeaf, UNFINISHED_APP
from app_tree import Leaf


def dfs_skeleton_picker(tree, old_skeleton):
    if isinstance(old_skeleton, App):
        assert isinstance(tree, App)
        maybe_new_fun = dfs_skeleton_picker(tree.fun, old_skeleton.fun)
        if maybe_new_fun is not None:
            return App(maybe_new_fun, old_skeleton.arg, None)

        maybe_new_arg = dfs_skeleton_picker(tree.arg, old_skeleton.arg)
        if maybe_new_arg is not None:
            return App(old_skeleton.fun, maybe_new_arg)

        return None

    if isinstance(old_skeleton, Leaf):
        return None

    if isinstance(old_skeleton, UnfinishedLeaf):
        if isinstance(tree, Leaf):
            return tree
        if isinstance(tree, App):
            return UNFINISHED_APP
        assert False

    assert False


def nested_mc_search(uf_tree, level, gen, eval_score, k, typ, skeleton_picker_strategy):
    if level == 0:
        tree = gen.gen_one_uf(uf_tree, k, typ)
        tree.score = eval_score(tree)
        return tree

    best = None
    while uf_tree.is_unfinished():
        for uf_successor in uf_tree.successors(gen, k, typ):
            tree = nested_mc_search(uf_successor, level - 1, gen, eval_score, k, typ)
            if best is None:
                best = tree
            else:
                best = tree if tree.score > best.score else best
        assert best is not None
        assert not best.is_unfinished()

        assert uf_tree.is_skeleton_of(best)
        uf_tree_new = skeleton_picker_strategy(best, uf_tree)
        assert uf_tree_new is not None
        assert uf_tree_new.is_skeleton_of(best)
        assert uf_tree.count_symbols() + 1 == uf_tree_new.count_symbols()
        uf_tree = uf_tree_new

    return best

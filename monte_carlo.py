
def nested_mc_search(level, gen, eval_score, unfin_tree):
    if level == 0:
        tree = gen.gen_one_unfin(unfin_tree)
        tree.score = eval_score(tree)
        return tree
    best = None
    while unfin_tree.is_unfinished():
        for successor in unfin_tree.successors():
            tree = nested_mc_search(level-1, gen, eval_score, successor)
            if best is None:
                best = tree
            else:
                best = tree if tree.score > best.score else best
        unfin_tree = best.take_prefix(unfin_tree.len)
    return best

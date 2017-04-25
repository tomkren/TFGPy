
def nested_mc_search(level, gen, eval_score, uf_tree):
    if level == 0:
        tree = gen.gen_one_uf(uf_tree)
        tree.score = eval_score(tree)
        return tree
    best = None
    while uf_tree.is_unfinished():
        for successor in uf_tree.successors(gen):
            tree = nested_mc_search(level-1, gen, eval_score, successor)
            if best is None:
                best = tree
            else:
                best = tree if tree.score > best.score else best

        # TODO
        uf_tree = best.take_prefix(uf_tree.len)
    return best

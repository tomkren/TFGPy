
def nested_mc_search(level, gen, eval_score, uf_tree, k, typ):
    if level == 0:
        tree = gen.gen_one_uf(uf_tree, k, typ)
        tree.score = eval_score(tree)
        return tree

    best = None
    while uf_tree.is_unfinished():
        for successor in uf_tree.successors(gen, k, typ):
            tree = nested_mc_search(level-1, gen, eval_score, successor, k, typ)
            if best is None:
                best = tree
            else:
                best = tree if tree.score > best.score else best

        # TODO
        uf_tree = best.take_prefix(uf_tree.len)
    return best

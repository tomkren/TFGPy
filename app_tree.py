from sub import mgu, Sub, dot
from typ import fresh, is_fun_type, split_fun_type, new_var, TypTerm


class AppTree:
    def is_well_typed(self, gamma):
        is_ok, _ = self.is_well_typed_acc(gamma, 0)
        return is_ok

    def is_well_typed_acc(self, gamma, n):
        raise NotImplementedError

    def apply_sub(self, sub):
        raise NotImplementedError

    def __eq__(self, other):
        if self is other:
            return True
        if type(other) != type(self):
            return False
        return self._eq_content(other)

    def _eq_content(self, other):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError

    def successors(self, gen, k, typ):
        ret = []
        for naive_succ in self.successors_naive(gen.gamma):
            num = gen.get_num_uf(naive_succ, k, typ)
            if num > 0:
                ret.append(naive_succ)
        return ret

    def successors_smart(self, gen, k):
        ret = []
        for uf_tree, sigma, n in self.successors_typed(gen.gamma, 0):
            num = gen.get_num_uf_smart(uf_tree, k)
            if num > 0:
                ret.append(uf_tree)
        return ret

    def successors_naive(self, gamma):
        raise NotImplementedError

    def successors_typed(self, gamma, n):
        raise NotImplementedError

    def get_unfinished_leafs(self):
        raise NotImplementedError

    def replace_unfinished_leafs(self, new_subtrees):
        acc = list(new_subtrees)
        ret = self.replace_unfinished_leafs_raw(acc)
        assert len(acc) == 0
        return ret

    def replace_unfinished_leafs_raw(self, new_subtrees):
        raise NotImplementedError

    def is_skeleton_of(self, tree):
        raise NotImplementedError

    def count_finished_nodes(self):
        raise NotImplementedError

    def is_finished(self):
        raise NotImplementedError

    def eval_str(self):
        return str(self)


class App(AppTree):
    def __init__(self, fun, arg, typ=None):
        assert fun.typ is None or is_fun_type(fun.typ)
        self.fun = fun
        self.arg = arg
        self.typ = typ

    def __repr__(self):
        return "App(%s, %s, %s)" % (repr(self.fun), repr(self.arg), repr(self.typ))

    def _eq_content(self, other):
        return self.fun == other.fun and self.arg == other.arg

    def __hash__(self):
        return 31 * hash(self.fun) + hash(self.arg)

    def __str__(self):
        return "(%s %s)" % (self.fun, self.arg)

    def eval_str(self):
        return "%s(%s)" % (self.fun.eval_str(), self.arg.eval_str())

    def apply_sub(self, sub):
        return App(self.fun.apply_sub(sub), self.arg.apply_sub(sub), sub(self.typ))

    def is_well_typed_acc(self, gamma, n):
        left, right = split_fun_type(self.fun.typ)

        if left == self.arg.typ and right == self.typ:
            is_ok, n1 = self.fun.is_well_typed_acc(gamma, n)
            if is_ok:
                return self.arg.is_well_typed_acc(gamma, n1)

        return False, None

    def successors_naive(self, gamma):
        fun_s = self.fun.successors_naive(gamma)
        if fun_s:
            return [App(fs, self.arg) for fs in fun_s]
        return [App(self.fun, ass) for ass in self.arg.successors_naive(gamma)]

    def successors_typed(self, gamma, n):
        n = self.typ.get_next_var_id(n)
        ret = []
        f_succs = self.fun.successors_typed(gamma, n)
        if f_succs:
            for tree_f, sub_f, n1 in f_succs:
                tree_x = self.arg.apply_sub(sub_f)
                ret.append((App(tree_f, tree_x, sub_f(self.typ)), sub_f, n1))
        else:
            x_succs = self.arg.successors_typed(gamma, n)
            for tree_x, sub_x, n1 in x_succs:
                tree_f = self.fun.apply_sub(sub_x)
                ret.append((App(tree_f, tree_x, sub_x(self.typ)), sub_x, n1))
        return ret

    def get_unfinished_leafs(self):
        return self.fun.get_unfinished_leafs().extend(self.arg.get_unfinished_leafs())

    def replace_unfinished_leafs_raw(self, new_subtrees):
        # možnost 1) zunifikovat minulej typ s tim novym a něco jak zeman
        # option (2) vracet substituci jak mazák
        # Any other option ?
        fun_new, fun_sub = self.fun.replace_unfinished_leafs(new_subtrees)
        arg_new, arg_sub = self.arg.replace_unfinished_leafs(new_subtrees)
        sigma = dot(arg_sub, fun_sub)
        return App(fun_new, arg_new, sigma(self.typ)), sigma

    def is_skeleton_of(self, tree):
        return (isinstance(tree, UnfinishedLeaf)
                or (isinstance(tree, App)
                    and self.fun.is_skeleton_of(tree.fun)
                    and self.arg.is_skeleton_of(tree.arg)))

    def count_finished_nodes(self):
        return 1 + self.fun.count_finished_nodes() + self.arg.count_finished_nodes()

    def is_finished(self):
        return self.fun.is_finished() and self.arg.is_finished()


class Leaf(AppTree):
    def __init__(self, sym, typ=None):
        self.sym = sym
        self.typ = typ

    def __repr__(self):
        return "Leaf(%s, %s)" % (repr(self.sym), repr(self.typ))

    def __str__(self):
        return str(self.sym)

    def _eq_content(self, other):
        return self.sym == other.sym

    def __hash__(self):
        return hash(self.sym)

    def apply_sub(self, sub):
        return Leaf(self.sym, sub(self.typ))

    def is_well_typed_acc(self, gamma, n):
        if self.sym not in gamma.ctx:
            return False, None

        t_s = gamma.ctx[self.sym].typ
        fr = fresh(t_s, self.typ, n)
        mu = mgu(self.typ, fr.typ)

        return not mu.is_failed(), fr.n

    def successors_naive(self, gamma):
        return []

    def successors_typed(self, gamma, n):
        return []

    def get_unfinished_leafs(self):
        return []

    def replace_unfinished_leafs_raw(self, new_subtrees):
        return self, Sub()

    def is_skeleton_of(self, tree):
        return (isinstance(tree, UnfinishedLeaf)
                or (isinstance(tree, Leaf)
                    and self.sym == tree.sym))

    def count_finished_nodes(self):
        return 1

    def is_finished(self):
        return True


class UnfinishedLeaf(Leaf):
    def __init__(self, typ=None):
        super().__init__("?", typ)

    def __repr__(self):
        return "UnfinishedLeaf(%s)" % repr(self.typ)

    def apply_sub(self, sub):
        raise NotImplementedError

    def is_well_typed_acc(self, gamma, n):
        return False, None

    def successors_naive(self, gamma):
        ret = [UNFINISHED_APP]
        for ctx_declaration in gamma.ctx.values():
            ret.append(Leaf(ctx_declaration.sym))
        return ret

    def successors_typed(self, gamma, n):
        alpha, n1 = new_var(self.typ, n)
        typ_f = TypTerm.make_arrow(alpha, self.typ)
        ret = [(App(UnfinishedLeaf(typ_f), UnfinishedLeaf(alpha), self.typ), Sub(), n1)]
        for ctx_declaration in gamma.ctx.values():
            fresh_res = fresh(ctx_declaration.typ, self.typ, n)
            mu = mgu(self.typ, fresh_res.typ)
            if not mu.is_failed():
                sigma = mu.restrict(self.typ)
                leaf = Leaf(ctx_declaration.sym, sigma(self.typ))
                ret.append((leaf, sigma, fresh_res.n))
        return ret

    def get_unfinished_leafs(self):
        return [self]

    def replace_unfinished_leafs_raw(self, new_subtrees):
        assert len(new_subtrees) > 0

        subtree = new_subtrees.pop(0)

        mu = mgu(self.typ, subtree.typ)

        assert not mu.is_failed()

        return subtree, mu

    def is_skeleton_of(self, tree):
        return True

    def count_finished_nodes(self):
        return 0

    def is_finished(self):
        return False


UNFINISHED_APP = App(UnfinishedLeaf(), UnfinishedLeaf())

INTERNAL_PAIR_CONSTRUCTOR_SYM = '_p_'


def split_internal_pair(tree):
    if isinstance(tree, App) and isinstance(tree.fun, App) and isinstance(tree.fun.fun, Leaf):
        if tree.fun.fun.sym == INTERNAL_PAIR_CONSTRUCTOR_SYM:
            return tree.arg, tree.fun.arg
    return None, None


def split_internal_tuple(tree):
    ret = []
    acc = tree
    while True:
        head, tail = split_internal_pair(acc)
        if head is None:
            ret.append(acc)
            return ret
        ret.append(head)
        acc = tail

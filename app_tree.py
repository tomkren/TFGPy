from itertools import chain

import sub
from typ import fresh, is_fun_type, split_fun_type, new_var, TypTerm, INTERNAL_PAIR_CONSTRUCTOR_SYM, T_INTERNAL_PAIR_CONSTRUCTOR


class AppTree:
    def __init__(self):
        self.counts = None

    def is_well_typed(self, gamma):
        is_ok, _ = self.is_well_typed_acc(gamma, {}, 0)
        return is_ok

    def is_well_typed_acc(self, gamma, local_vars, n):
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

    def lower_size_bound(self):
        raise NotImplementedError

    def size(self):
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
        counts = self.count_nodes()
        return counts[App] + counts[Leaf] - counts[UnfinishedLeaf]

    def count_nodes(self):
        if self.counts is None:
            self.counts = self.count_nodes_raw()
        return self.counts

    def count_nodes_raw(self):
        return {type(self): 1}

    def is_finished(self):
        counts = self.count_nodes()
        return UnfinishedLeaf not in counts

    def eval_str(self):
        return str(self)

    def map_reduce(self, mapper, reducer):
        return mapper(self)


class Abs(AppTree):
    def __init__(self, var, body, typ=None):
        super().__init__()
        assert isinstance(var, Var)
        # assert isinstance(body, AppTree)
        assert typ is None or is_fun_type(typ)
        self.var = var  # TODO nebo radši držet jen var_id ????????????????????????????
        self.body = body
        self.typ = typ

    def __repr__(self):
        return "App(%s, %s, %s)" % (repr(self.var), repr(self.body), repr(self.typ))

    def _eq_content(self, other):
        return self.var == other.var and self.body == other.body

    def __hash__(self):
        return 31 * hash(self.var) + hash(self.body)

    def __str__(self):
        return "(λ%s.%s)" % (self.var, self.body)

    def eval_str(self):
        return "(lambda %s: %s)" % (self.var.eval_str(), self.body.eval_str())

    def apply_sub(self, sub):
        return Abs(self.var.apply_sub(sub), self.body.apply_sub(sub), sub(self.typ))

    def is_well_typed_acc(self, gamma, local_vars, n):
        left, right = split_fun_type(self.typ)
        var_typ = self.var.typ
        if left == var_typ and right == self.body.typ:

            # For efficiency local_vars is modified (instead of copied).
            var_id = self.var.var_id()
            old_var_typ = local_vars.get(var_id)
            local_vars[var_id] = var_typ

            is_ok, n1 = self.body.is_well_typed_acc(gamma, local_vars, n)

            # Undo local_vars modification.
            if old_var_typ is None:
                del local_vars[var_id]
            else:
                local_vars[var_id] = old_var_typ

            return is_ok, n1

        return False, None

    def successors_naive(self, gamma):
        return [Abs(self.var, body2) for body2 in self.body.successors_naive(gamma)]

    def successors_typed(self, gamma, n):
        n = self.typ.get_next_var_id(n)
        ret = []
        body_succs = self.body.successors_typed(gamma, n)
        for body_new, sigma, n_new in body_succs:
            var_new = self.var.apply_sub(sigma)
            ret.append((Abs(var_new, body_new, sigma(self.typ)), sigma, n_new))
        return ret

    def get_unfinished_leafs(self):
        return self.body.get_unfinished_leafs()

    def lower_size_bound(self):
        return 1 + self.body.lower_size_bound()

    def size(self):
        return 1 + self.body.size()

    def replace_unfinished_leafs_raw(self, new_subtrees):
        body_new, sigma = self.body.replace_unfinished_leafs_raw(new_subtrees)
        var_new = self.var.apply_sub(sigma)
        return Abs(var_new, body_new, sigma(self.typ)), sigma

    # TODO revidovat
    # note: pozor, bere v potaz jen netypovaný (dumb) skeletony
    def is_skeleton_of(self, tree):
        return (isinstance(tree, UnfinishedLeaf)
                or (isinstance(tree, Abs) and self.body.is_skeleton_of(tree.body)))

    # TODO revidovat
    def count_nodes_raw(self):
        counts = {type(self): 1}
        for t, c in self.body.count_nodes().items():
            counts[t] = counts.get(t, 0) + c
        return counts

    # TODO revidovat
    def map_reduce(self, mapper, reducer):
        a = mapper(self)
        b = self.body.map_reduce(mapper, reducer)
        return reducer(a, b)


class App(AppTree):
    def __init__(self, fun, arg, typ=None):
        super().__init__()
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

    def is_well_typed_acc(self, gamma, local_vars, n):
        left, right = split_fun_type(self.fun.typ)

        if left == self.arg.typ and right == self.typ:
            is_ok, n1 = self.fun.is_well_typed_acc(gamma, local_vars, n)
            if is_ok:
                return self.arg.is_well_typed_acc(gamma, local_vars, n1)

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
        ufs1 = self.fun.get_unfinished_leafs()
        ufs2 = self.arg.get_unfinished_leafs()
        ufs1.extend(ufs2)
        return ufs1

    def lower_size_bound(self):
        return self.fun.lower_size_bound() + self.arg.lower_size_bound()

    def size(self):
        return self.fun.size() + self.arg.size()

    def replace_unfinished_leafs_raw(self, new_subtrees):
        fun_new1, fun_sub = self.fun.replace_unfinished_leafs_raw(new_subtrees)
        arg_new1 = self.arg.apply_sub(fun_sub)
        arg_new2, arg_sub = arg_new1.replace_unfinished_leafs_raw(new_subtrees)
        fun_new2 = fun_new1.apply_sub(arg_sub)
        sigma = sub.dot(arg_sub, fun_sub)
        return App(fun_new2, arg_new2, sigma(self.typ)), sigma

    # TODO revidovat
    # note: pozor, bere v potaz jen netypovaný (dumb) skeletony
    def is_skeleton_of(self, tree):
        return (isinstance(tree, UnfinishedLeaf)
                or (isinstance(tree, App)
                    and self.fun.is_skeleton_of(tree.fun)
                    and self.arg.is_skeleton_of(tree.arg)))

    # TODO revidovat
    def count_nodes_raw(self):
        counts = {type(self): 1}
        for t, c in chain(self.fun.count_nodes().items(), self.arg.count_nodes().items()):
            counts[t] = counts.get(t, 0) + c
        return counts

    # TODO revidovat
    def map_reduce(self, mapper, reducer):
        a = mapper(self)
        b = self.fun.map_reduce(mapper, reducer)
        c = self.arg.map_reduce(mapper, reducer)
        return reducer(reducer(a, b), c)


class Leaf(AppTree):
    def __init__(self, sym, typ=None):
        super().__init__()
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

    def is_well_typed_acc(self, gamma, local_vars, n):

        if self.sym == INTERNAL_PAIR_CONSTRUCTOR_SYM:
            t_s = T_INTERNAL_PAIR_CONSTRUCTOR
        else:
            if self.sym not in gamma.ctx:
                return False, None
            t_s = gamma.ctx[self.sym].typ

        fr = fresh(t_s, self.typ, n)
        mu = sub.mgu(self.typ, fr.typ)

        return not mu.is_failed(), fr.n

    def successors_naive(self, gamma):
        return []

    def successors_typed(self, gamma, n):
        return []

    def get_unfinished_leafs(self):
        return []

    def lower_size_bound(self):
        return 1

    def size(self):
        return 1

    def replace_unfinished_leafs_raw(self, new_subtrees):
        return self, sub.Sub()

    # TODO revidovat
    def is_skeleton_of(self, tree):
        return (isinstance(tree, UnfinishedLeaf)
                or (isinstance(tree, Leaf)
                    and self.sym == tree.sym))


class Var(Leaf):
    def __init__(self, var_id, typ=None):
        super().__init__(var_id, typ)

    def var_id(self):
        return self.sym

    def apply_sub(self, sub):
        return Var(self.var_id(), sub(self.typ))

    def __repr__(self):
        return "Var(%s, %s)" % (repr(self.var_id()), repr(self.typ))

    def __str__(self):
        return 'x' + str(self.var_id())

    def is_well_typed_acc(self, gamma, local_vars, n):
        var_id = self.var_id()
        if var_id in local_vars:
            if self.typ == local_vars[var_id]:
                return True, self.typ.get_next_var_id(n)
        return False, None

    # TODO revidovat
    def is_skeleton_of(self, tree):
        return (isinstance(tree, UnfinishedLeaf)
                or (isinstance(tree, Var)
                    and self.var_id() == tree.var_id()))


class UnfinishedLeaf(Leaf):
    def __init__(self, typ=None):
        super().__init__("?", typ)

    def __repr__(self):
        return "UnfinishedLeaf(%s)" % repr(self.typ)

    # def __str__(self):
    #    typ_str = '<'+str(self.typ)+'>' if self.typ else ''
    #    return str(self.sym + typ_str)

    def apply_sub(self, sub):
        # raise NotImplementedError
        return UnfinishedLeaf(sub(self.typ))

    def is_well_typed_acc(self, gamma, local_vars, n):
        return False, None

    def successors_naive(self, gamma):
        ret = [UNFINISHED_APP]
        for ctx_declaration in gamma.ctx.values():
            ret.append(Leaf(ctx_declaration.sym))
        return ret

    def successors_typed(self, gamma, n):
        alpha, n1 = new_var(self.typ, n)
        typ_f = TypTerm.make_arrow(alpha, self.typ)
        ret = [(App(UnfinishedLeaf(typ_f), UnfinishedLeaf(alpha), self.typ), sub.Sub(), n1)]
        for ctx_declaration in gamma.ctx.values():
            fresh_res = fresh(ctx_declaration.typ, self.typ, n)
            mu = sub.mgu(self.typ, fresh_res.typ)
            if not mu.is_failed():
                sigma = mu.restrict(self.typ)
                leaf = Leaf(ctx_declaration.sym, sigma(self.typ))
                ret.append((leaf, sigma, fresh_res.n))
        return ret

    def get_unfinished_leafs(self):
        return [self]

    def lower_size_bound(self):
        return 1

    def size(self):
        return 0

    def replace_unfinished_leafs_raw(self, new_subtrees):
        assert len(new_subtrees) > 0
        subtree = new_subtrees.pop(0)
        mu = sub.mgu(self.typ, subtree.typ)
        assert not mu.is_failed()
        new_subtree = subtree.apply_sub(mu)
        return new_subtree, mu

    def is_skeleton_of(self, tree):
        return True


UNFINISHED_APP = App(UnfinishedLeaf(), UnfinishedLeaf())


def split_internal_pair(tree):
    if isinstance(tree, App) and isinstance(tree.fun, App) and isinstance(tree.fun.fun, Leaf):
        if tree.fun.fun.sym == INTERNAL_PAIR_CONSTRUCTOR_SYM:
            return tree.fun.arg, tree.arg
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

from sub import mgu
from typ import fresh, is_fun_type, split_fun_type


class AppTree:
    def __init__(self):
        self.finished_flag = None

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

    def successors_naive(self, gamma):
        raise NotImplementedError

    def is_skeleton_of(self, tree):
        raise NotImplementedError

    def count_finished_nodes(self):
        raise NotImplementedError

    def is_finished(self):
        if self.finished_flag is None:
            self.finished_flag = self.is_finished_raw()
        return self.finished_flag

    def is_finished_raw(self):
        raise NotImplementedError

    def eval_str(self):
        return str(self)


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

    def is_skeleton_of(self, tree):
        return (isinstance(tree, UnfinishedLeaf)
                or (isinstance(tree, App)
                    and self.fun.is_skeleton_of(tree.fun)
                    and self.arg.is_skeleton_of(tree.arg)))

    def count_finished_nodes(self):
        return 1 + self.fun.count_finished_nodes() + self.arg.count_finished_nodes()

    def is_finished_raw(self):
        return self.fun.is_finished_raw() and self.arg.is_finished_raw()


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

    def is_well_typed_acc(self, gamma, n):
        if self.sym not in gamma.ctx:
            return False, None

        t_s = gamma.ctx[self.sym].typ
        fr = fresh(t_s, self.typ, n)
        mu = mgu(self.typ, fr.typ)

        return not mu.is_failed(), fr.n

    def successors_naive(self, gamma):
        return []

    def is_skeleton_of(self, tree):
        return (isinstance(tree, UnfinishedLeaf)
                or (isinstance(tree, Leaf)
                    and self.sym == tree.sym))

    def count_finished_nodes(self):
        return 1

    def is_finished_raw(self):
        return True


class UnfinishedLeaf(Leaf):
    def __init__(self):
        super().__init__("?")

    def __repr__(self):
        return "UnfinishedLeaf()"

    def apply_sub(self, sub):
        raise NotImplementedError

    def is_well_typed_acc(self, gamma, n):
        return False, None

    def successors_naive(self, gamma):
        ret = [UNFINISHED_APP]
        for ctx_declaration in gamma.ctx.values():
            ret.append(Leaf(ctx_declaration.sym))
        return ret

    def is_skeleton_of(self, tree):
        return True

    def count_finished_nodes(self):
        return 0

    def is_finished_raw(self):
        return False


UNFINISHED_APP = App(UnfinishedLeaf(), UnfinishedLeaf())

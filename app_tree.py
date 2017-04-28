from typing import List

from sub import mgu
from typ import fresh, is_fun_type, split_fun_type


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

    def successors_naive(self, gamma):
        raise NotImplementedError


class App(AppTree):
    def __init__(self, fun, arg, typ):
        assert is_fun_type(fun.typ)
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
            return [App(fs, self.arg, None) for fs in fun_s]
        return [App(self.fun, ass, None) for ass in self.arg.successors_naive(gamma)]


class Leaf(AppTree):
    def __init__(self, sym, typ):
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


class UnfinishedLeaf(Leaf):
    def __init__(self):
        super().__init__("?", None)

    def __repr__(self):
        return "UnfinishedLeaf()"

    def apply_sub(self, sub):
        raise NotImplementedError

    def is_well_typed_acc(self, gamma, n):
        return False, None

    def successors_naive(self, gamma):
        ret = [App(UnfinishedLeaf(), UnfinishedLeaf(), None)]
        for ctx_declaration in gamma.values():
            ret.append(Leaf(ctx_declaration.sym, None))
        return ret

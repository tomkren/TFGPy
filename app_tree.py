from generator import ts1_static
from sub import mgu
from typ import fresh, is_fun_type, split_fun_type, TypTerm


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

    def successors(self, gen):
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

    def successors(self, gen):
        fun_s = self.fun.successors(gen)
        if fun_s:
            ret = []
            for fs in fun_s:
                # TODO fs by mela byt pekna datova strukturka
                ret.append(
                    App(fs.fun, self.arg, self.typ.apply_sub(fs.sub))
                )

        s_arg = self.arg.successors(gen)


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

    def successors(self, gen):
        return []


class UnfinishedLeaf(Leaf):
    def __init__(self, typ, k):
        super().__init__("?", typ)
        # self.typ = typ
        assert k > 0
        self.k = k

    def __repr__(self):
        return "UnfinishedLeaf(%s, %s)" % (repr(self.typ), self.k)

    def __str__(self):
        return "%d: %s" % (self.k, self.typ)

    def _eq_content(self, other):
        return self.typ == other.typ and self.k == other.k

    def __hash__(self):
        return hash(self.typ) + 31 * hash(self.k)

    def apply_sub(self, sub):
        raise NotImplementedError

    def is_well_typed_acc(self, gamma, n):
        return False, None

    def successors(self, gen):
        if self.k == 1:
            pre_ts1_results = ts1_static(gen.gamma, self.typ, 0)
            return [Leaf(tr.sym, tr.sub(self.typ)) for tr in pre_ts1_results]
        else:
            alpha = self.typ.get_next_var_id()
            ret = []
            for i in range(1, self.k):
                j = self.k - i
                fun = UnfinishedLeaf(TypTerm.make_arrow(alpha, self.typ), i)
                arg = UnfinishedLeaf(alpha, j)
                app = App(fun, arg, self.typ)
                # TODO check this app out
                ret.append(app)
            return ret

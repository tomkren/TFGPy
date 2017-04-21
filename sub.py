import copy
from collections import namedtuple

import utils
from typ import Typ, TypSymbol, TypVar, TypTerm

PreTs1Res = namedtuple('PreTs1Res', ['sym', 'sub'])
Ts1Res = namedtuple('Ts1Res', ['sym', 'sub', 'n'])

# PreTsRes = namedtuple('PreTsRes', ['tree', 'sub'])
TsRes = namedtuple('TsRes', ['tree', 'sub', 'n'])

PreSubRes = namedtuple('PreSubRes', ['num', 'sub'])
SubRes = namedtuple('SubRes', ['num', 'sub', 'n'])

MoverRes = namedtuple('MoverRes', ['sub', 'n'])


def mgu(t1: Typ, t2: Typ):
    # TODO: possibly optimize by removing the non-local
    # variables

    table = {}
    fail = None
    agenda = [(t1, t2)]

    def mgu_process_var(var, typ):
        nonlocal fail, table, agenda
        if not isinstance(typ, Typ):
            fail = "Not a Typ: '%s'" % (typ)
            return
        if typ.contains_var(var):
            fail = "Occur check fail: '%s' in '%s'" % (var, typ)
            return
        if var == typ:
            return

        for key, val in table.items():
            table[key] = val.apply_mini_sub(var, typ)

        agenda = [(f.apply_mini_sub(var, typ),
                   s.apply_mini_sub(var, typ)) for f, s in agenda]

        table[var] = typ

    def mgu_process_term(term1, term2):
        nonlocal fail, agenda

        if len(term1.arguments) != len(term2.arguments):
            fail = "Term length mismatch: '%s' and '%s'" % (term1, term2)
            return

        agenda.extend(zip(term1.arguments, term2.arguments))

    def mgu_process(a1, a2):
        nonlocal fail
        if a1 == a2:
            return

        if isinstance(a1, TypSymbol) and isinstance(a2, TypSymbol):
            fail = "Symbols mismatch: '%s' and '%s'" % (a1, a2)
            return

        if isinstance(a1, TypVar):
            mgu_process_var(a1, a2)
            return

        if isinstance(a2, TypVar):
            mgu_process_var(a2, a1)
            return

        if isinstance(a1, TypTerm) and isinstance(a2, TypTerm):
            mgu_process_term(a1, a2)
            return

        if ((isinstance(a1, TypTerm) and isinstance(a2, TypSymbol)) or
                (isinstance(a2, TypTerm) and isinstance(a1, TypSymbol))):
            fail = "Kind mismatch: '%s' and '%s'" % (a1, a2)
            return

        fail = "Not Typs: '%s' and '%s'" % (a1, a2)

    while agenda and not fail:
        a1, a2 = agenda.pop()
        mgu_process(a1, a2)

    return Sub(table, fail)


class Sub:
    def __init__(self, table=None, fail_msg=None):
        if table is None:
            table = {}
        self.table = table
        self.fail_msg = fail_msg

    def __call__(self, typ):
        return typ.apply_sub(self)

    def __hash__(self):
        if self.fail_msg is not None:
            raise RuntimeError("Hashing failed sub.")

        return hash(tuple(sorted(repr(it) for it in self.table.items())))

    def restrict(self, typ):
        if self.is_failed():
            raise RuntimeError("Restrict on a failed %s" % repr(self))
        keyset = typ.get_sub_keys()
        new_table = {}
        for k, v in self.table.items():
            if k in keyset:
                new_table[k] = v

        return Sub(new_table)

    def domain(self):
        return set(self.table.keys())

    def is_failed(self):
        return bool(self.fail_msg)

    def __eq__(self, other):
        if not self.fail_msg:
            return self.table == other.table

        raise RuntimeError("Comparing two failed substitutions.")

    def __repr__(self):
        if not self.fail_msg:
            return "Sub(%s)" % (self.table)
        else:
            return "Sub(fail_msg='%s')" % (self.fail_msg)

    def __str__(self):
        if not self.fail_msg:
            return "\n".join("%s : %s" % (k, v) for k, v in self.table.items())
        else:
            return "FAIL: %s" % (self.fail_msg)


def dot(g, f):
    ret = copy.copy(g.table)
    for f_key, f_val in f.table.items():
        gf_val = g(f_val)
        if gf_val == f_key:
            del ret[f_key]
        else:
            ret[f_key] = gf_val

    return Sub(ret)


class Mover:
    def __init__(self, typ: Typ, n):
        self.typ = typ
        self.tnvi_0 = typ.get_next_var_id(0)
        self.tnvi_n = max(self.tnvi_0, n)

    def is_move_needed(self):
        return self.tnvi_n > self.tnvi_0

    def make_delta(self, sub):
        codomain_vars = utils.update_union((t.get_vars() for t in sub.table.values()), set())

        delta_table = {}
        nvi = self.tnvi_n

        for var in codomain_vars:
            if not isinstance(var.name, int):
                continue
            if var.name >= self.tnvi_0:
                delta_table[var] = TypVar(nvi)
                nvi += 1

        return Sub(delta_table), nvi

    def move_sub(self, sub):
        delta, nvi = self.make_delta(sub)

        moved_sub = dot(delta, sub).restrict(self.typ)
        return MoverRes(moved_sub, nvi)

    def move_sub_n_tree(self, sub, tree):
        delta, nvi = self.make_delta(sub)

        moved_sub = dot(delta, sub).restrict(self.typ)
        moved_tree = tree.apply_sub(delta)

        return MoverRes(moved_sub, nvi), moved_tree

    @staticmethod
    def move_results(typ: Typ, n, results, zipper):
        m = Mover(typ, n)
        return [zipper(res, m.move_sub(res.sub)) for res in results]

    @staticmethod
    def move_pre_sub_results(typ, n, pre_sub_results):
        return Mover.move_results(typ, n, pre_sub_results, (
            lambda psr, mr: SubRes(psr.num, mr.sub, mr.n)
        ))

    @staticmethod
    def move_pre_ts1_results(typ, n, pre_ts1_results):
        return Mover.move_results(typ, n, pre_ts1_results, (
            lambda ts1r, mr: Ts1Res(ts1r.sym, mr.sub, mr.n)
        ))

    @staticmethod
    def move_sub_results(typ, n, sub_results):
        return Mover.move_results(typ, n, sub_results, (
            lambda sub_res, mr: SubRes(sub_res.num, mr.sub, mr.n)
        ))

    @staticmethod
    def move_ts_results(typ, n, ts_results):
        m = Mover(typ, n)
        ret = []

        for res in ts_results:
            mr, mtree = m.move_sub_n_tree(res.sub, res.tree)
            ret.append(TsRes(mtree, mr.sub, mr.n))

        return ret

    @staticmethod
    def move_sub_results_0(typ, n, results):
        m = Mover(typ, n)
        if not m.is_move_needed():
            return results

        ret = []
        for res in results:
            mr = m.move_sub(res.sub)
            ret.append(SubRes(res.num, mr.sub, mr.n))
        return ret

    # NOTE possibly refactor into zipper
    @staticmethod
    def move_ts1_results_0(typ, n, results):
        m = Mover(typ, n)
        if not m.is_move_needed():
            return results

        ret = []
        for res in results:
            mr = m.move_sub(res.sub)
            ret.append(Ts1Res(res.sym, mr.sub, mr.n))
        return ret


if __name__ == "__main__":
    pass

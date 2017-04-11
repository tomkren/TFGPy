from collections import namedtuple

from context import Context
from sub import mgu
from typ import Typ, fresh, new_var

PreTs1Res = namedtuple('PreTs1Res', ['sym', 'sub'])
PreSubRes = namedtuple('PreSubRes', ['num', 'sigma'])
SubRes = namedtuple('SubRes', ['num', 'sigma', 'n'])


def ts1_static(gamma: Context, typ: Typ, n):
    ret = []
    for sym in gamma.ctx.values():
        f = fresh(sym.typ, typ, n)
        mu = mgu(typ, f.typ)
        if not mu.is_failed():
            sigma = mu.restrict(typ)
            ret.append(PreTs1Res(sym, sigma))

    return ret


def pack(typ, n, pre_sub_results):
    pass


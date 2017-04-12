from collections import OrderedDict
from collections import namedtuple

from context import Context
from sub import mgu, Mover, SubRes, PreTs1Res
from typ import Typ, fresh, new_var


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
    results = OrderedDict()
    sub_results = Mover.move_pre_sub_results(typ, n, pre_sub_results)

    for res in sub_results:
        sigma = res.sub
        val = results.get(sigma, None)
        if val is None:
            results[sigma] = res
        else:
            assert res.n == val.n
            results[sigma] = SubRes(val.num + res.num, val.sub, val.n)

    # NOTE: watchout, returning generator
    return results.values()

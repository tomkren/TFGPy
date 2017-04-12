import sub
from app_tree import Leaf, App
from generator import ts1_static
from sub import Mover, TsRes

from typ import new_var
from typ_utils import mk_fun_type


def ts_1(gamma, typ, n):
    pre_ts1_res = ts1_static(gamma, typ, n)
    ts1_res = Mover.move_pre_ts1_results(typ, n, pre_ts1_res)

    return [TsRes(tree=Leaf(tr.sym, tr.sub(typ)),
                  sub=tr.sub,
                  n=tr.n) for tr in ts1_res]


def ts_ij(gamma, i, j, typ, n):
    ret = []
    alpha, n1 = new_var(typ, n)
    typ_f = mk_fun_type(alpha, typ)

    for res_f in ts(gamma, i, typ_f, n1):
        typ_x = res_f.sub(alpha)
        for res_x in ts(gamma, j, typ_x, res_f.n):
            sigma_fx = sub.dot(res_x.sub, res_f.sub).restrict(typ)
            tree_fx = App(res_f.tree, res_x.tree, sigma_fx(typ))
            ret.append(TsRes(tree_fx, sigma_fx, res_x.n))

    return ret


def ts(gamma, k, typ, n):
    assert k >= 1
    if k == 1:
        return ts_1(gamma, typ, n)

    ret = [ts_ij(gamma, i, k - i, typ, n)
           for i in range(1, k - 1)]
    return Mover.move_ts_results(typ, n, ret)

import sub
from app_tree import Leaf, App, UnfinishedLeaf, AppTree
from generator import ts1_static, pack, subs_uf_sym
from sub import Mover, TsRes

from typ import new_var, TypTerm, fresh


def ts_1(gamma, typ, n):
    pre_ts1_res = ts1_static(gamma, typ, n)
    ts1_res = Mover.move_pre_ts1_results(typ, n, pre_ts1_res)

    ret = [TsRes(tree=Leaf(tr.sym, tr.sub(typ)),
                 sub=tr.sub,
                 n=tr.n) for tr in ts1_res]

    return ret


def ts_ij(gamma, i, j, typ, n):
    ret = []
    alpha, n1 = new_var(typ, n)
    typ_f = TypTerm.make_arrow(alpha, typ)

    for res_f in ts(gamma, i, typ_f, n1):
        typ_x = res_f.sub(alpha)
        for res_x in ts(gamma, j, typ_x, res_f.n):
            sigma_fx = sub.dot(res_x.sub, res_f.sub).restrict(typ)
            tree_f = res_f.tree.apply_sub(res_x.sub)
            tree_fx = App(tree_f, res_x.tree, sigma_fx(typ))
            ret.append(TsRes(tree_fx, sigma_fx, res_x.n))

    return ret


def ts(gamma, k, typ, n):
    assert k >= 1
    if k == 1:
        return ts_1(gamma, typ, n)

    ret = []
    for i in range(1, k):
        ret.extend(ts_ij(gamma, i, k - i, typ, n))

    return Mover.move_ts_results(typ, n, ret)


def subs_1(gamma, typ, n):
    pre_ts1_res = ts1_static(gamma, typ, n)
    pre_sub_results_unpacked = (sub.PreSubRes(1, res.sub) for res in pre_ts1_res)
    return pack(typ, n, pre_sub_results_unpacked)


def subs_ij(gamma, i, j, typ, n):
    ret = []
    alpha, n1 = new_var(typ, n)
    typ_f = TypTerm.make_arrow(alpha, typ)

    for res_f in subs(gamma, i, typ_f, n1):
        typ_x = res_f.sub(alpha)
        for res_x in subs(gamma, j, typ_x, res_f.n):
            sigma_fx = sub.dot(res_x.sub, res_f.sub).restrict(typ)
            num_fx = res_x.num * res_f.num

            ret.append(sub.PreSubRes(num_fx, sigma_fx))

    return ret


def subs_uf_ij(gamma, f_uf, x_uf, i, j, typ, n):
    ret = []
    alpha, n1 = new_var(typ, n)
    typ_f = TypTerm.make_arrow(alpha, typ)

    for res_f in subs_uf(gamma, f_uf, i, typ_f, n1):
        typ_x = res_f.sub(alpha)
        for res_x in subs_uf(gamma, x_uf, j, typ_x, res_f.n):
            sigma_fx = sub.dot(res_x.sub, res_f.sub).restrict(typ)
            num_fx = res_x.num * res_f.num

            ret.append(sub.PreSubRes(num_fx, sigma_fx))

    return ret


def subs(gamma, k, typ, n):
    assert k >= 1
    if k == 1:
        return subs_1(gamma, typ, n)

    ret = []
    for i in range(1, k):
        ret.extend(subs_ij(gamma, i, k - i, typ, n))

    return pack(typ, n, ret)


def subs_uf(gamma, uf_tree, k, typ, n):
    if isinstance(uf_tree, UnfinishedLeaf):
        return subs(gamma, k, typ, n)

    if isinstance(uf_tree, Leaf):
        return subs_uf_sym(gamma, n, typ, uf_tree)

    if isinstance(uf_tree, App):
        if k == 1:
            return []
        assert k > 1
        ret = []
        for i in range(1, k):
            ret.extend(subs_uf_ij(gamma, uf_tree.fun, uf_tree.arg, i, k - i, typ, n))
        return pack(typ, n, ret)

    assert False


def get_num(gamma, k, typ):
    return sum(res.num for res in subs(gamma, k, typ, 0))

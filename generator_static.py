import sub
from app_tree import Leaf, App
from generator import ts1_static
from sub import Mover, TsRes

from typ import new_var, TypTerm


def ts_1(gamma, typ, n):
    print("ts_1\t", n, "'%s'"%typ)
    pre_ts1_res = ts1_static(gamma, typ, n)
    ts1_res = Mover.move_pre_ts1_results(typ, n, pre_ts1_res)

    ret = [TsRes(tree=Leaf(tr.sym, tr.sub(typ)),
                 sub=tr.sub,
                 n=tr.n) for tr in ts1_res]

    return ret


def ts_ij(gamma, i, j, typ, n):
    print("ENTER ts_ij\t", i, j, n, "'%s'"%typ)

    nase = False
    if i==2 and j == 1 and n==0 and str(typ) == '((P A (P A A)) -> (P A (P A A)))':
        print("#"*20)
        nase = True
    ret = []
    alpha, n1 = new_var(typ, n)
    typ_f = TypTerm.make_arrow(alpha, typ)

    O, I = 0, 0
    for res_f in ts(gamma, i, typ_f, n1):
        O+=1
        print("res_f", O, I)
        typ_x = res_f.sub(alpha)
        print(res_f.sub)
        for res_x in ts(gamma, j, typ_x, res_f.n):
            I+=1
            print("res_x", O,I)
            print(res_x.sub)
            sigma_fx = sub.dot(res_x.sub, res_f.sub).restrict(typ)
            tree_f = res_f.tree.apply_sub(res_x.sub)
            tree_fx = App(tree_f, res_x.tree, sigma_fx(typ))
            if not tree_fx.is_well_typed(gamma):
                assert False
            ret.append(TsRes(tree_fx, sigma_fx, res_x.n))

    print("LEAVE ts_ij\t", i, j, n, "'%s'"%typ)
    return ret


def ts(gamma, k, typ, n):
    print("ts\t", k,n, "'%s'"%typ)

    assert k >= 1
    if k == 1:
        return ts_1(gamma, typ, n)

    ret = []
    for i in range(1, k):
        ret.extend(ts_ij(gamma, i, k - i, typ, n))

    return Mover.move_ts_results(typ, n, ret)

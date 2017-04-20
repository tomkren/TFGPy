from collections import OrderedDict

import sub
from cache import Cache
from context import Context
from normalization import Normalizator
from sub import mgu, Mover, SubRes, PreTs1Res, PreSubRes
from tracer_deco import tracer_deco
from typ import Typ, fresh, new_var, TypTerm


def ts1_static(gamma: Context, typ: Typ, n):
    ret = []
    for ctx_declaration in gamma.ctx.values():
        f = fresh(ctx_declaration.typ, typ, n)
        mu = mgu(typ, f.typ)
        if not mu.is_failed():
            sigma = mu.restrict(typ)
            ret.append(PreTs1Res(ctx_declaration.sym, sigma))
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

    return list(results.values())


class Generator:
    def __init__(self, gamma, cache=Cache, normalizator=Normalizator):
        self.gamma = gamma
        self.cache = cache(self)
        self.normalizator = normalizator

    def get_num(self, k, typ):
        nf = self.normalizator(typ)
        return self.cache.get_num(k, nf.typ_nf)

    def ts_1_compute(self, typ, n):
        pre_ts1_res = ts1_static(self.gamma, typ, n)
        return Mover.move_pre_ts1_results(typ, n, pre_ts1_res)

    @tracer_deco()
    def subs_ij(self, i, j, typ, n):
        ret = []
        alpha, n1 = new_var(typ, n)
        typ_f = TypTerm.make_arrow(alpha, typ)

        for res_f in self.subs(i, typ_f, n1):
            typ_x = res_f.sub(alpha)
            for res_x in self.subs(j, typ_x, res_f.n):
                sigma_fx = sub.dot(res_x.sub, res_f.sub).restrict(typ)
                num_fx = res_x.num * res_f.num

                ret.append(sub.PreSubRes(num_fx, sigma_fx))
        return ret

    @tracer_deco(log_ret=True, ret_pp=lambda results: "\n".join("NUM=%d\tN=%d\n%s"%(r.num, r.n, r.sub) for r in results))
    def subs(self, k, typ, n):
        nf = self.normalizator(typ)
        results_nf = self.cache.subs(k, nf.typ_nf, n)
        ret = nf.denormalize(results_nf, n)

        return ret

    @tracer_deco()
    def subs_compute(self, k, typ, n):
        assert k >= 1
        if k == 1:
            ret = (PreSubRes(1, res.sub) for res in self.cache.ts_1(typ, n))
        else:
            ret = []
            for i in range(1, k):
                ret.extend(self.subs_ij(i, k - i, typ, n))

        return pack(typ, n, ret)




import random
from collections import OrderedDict

import sub
from app_tree import Leaf, App, UnfinishedLeaf
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


def subs_uf_sym(gamma, n, typ, uf_tree):
    # TODO factor out uf_tree.sym to args
    ctx_declaration = gamma.ctx.get(uf_tree.sym, None)
    if ctx_declaration is None:
        return []
    f = fresh(ctx_declaration.typ, typ, n)
    mu = sub.mgu(typ, f.typ)
    if mu.is_failed():
        return []
    sigma = mu.restrict(typ)
    return [sub.SubRes(1, sigma, f.n)]


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

    def get_num_uf(self, uf_tree, k, typ):
        return sum(sd.num for sd in self.subs_uf(uf_tree, k, typ, 0))

    def ts_1_compute(self, typ, n):
        pre_ts1_res = ts1_static(self.gamma, typ, n)
        return Mover.move_pre_ts1_results(typ, n, pre_ts1_res)

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

    def subs_uf_ij(self, f_uf, x_uf, i, j, typ, n):
        ret = []
        alpha, n1 = new_var(typ, n)
        typ_f = TypTerm.make_arrow(alpha, typ)

        for res_f in self.subs_uf(f_uf, i, typ_f, n1):
            typ_x = res_f.sub(alpha)
            for res_x in self.subs_uf(x_uf, j, typ_x, res_f.n):
                sigma_fx = sub.dot(res_x.sub, res_f.sub).restrict(typ)
                num_fx = res_x.num * res_f.num

                ret.append(sub.PreSubRes(num_fx, sigma_fx))

        return ret

    #@tracer_deco(log_ret=True,
                 #ret_pp=lambda results: "\n".join("NUM=%d\tN=%d\n%s" % (r.num, r.n, r.sub) for r in results))
    def subs(self, k, typ, n):
        nf = self.normalizator(typ)
        results_nf = self.cache.subs(k, nf.typ_nf, n)
        ret = nf.denormalize(results_nf, n)

        return ret

    def subs_uf(self, uf_tree, k, typ, n):
        if isinstance(uf_tree, UnfinishedLeaf):
            return self.subs(k, typ, n)

        if isinstance(uf_tree, Leaf):
            if k == 1:
                return subs_uf_sym(self.gamma, n, typ, uf_tree)
            assert k > 1
            return []

        if isinstance(uf_tree, App):
            if k == 1:
                return []
            assert k > 1
            ret = []
            for i in range(1, k):
                ret.extend(self.subs_uf_ij(uf_tree.fun, uf_tree.arg, i, k - i, typ, n))
            return pack(typ, n, ret)

        assert False

    #@tracer_deco()
    def subs_compute(self, k, typ, n):
        assert k >= 1
        if k == 1:
            ret = (PreSubRes(1, res.sub) for res in self.cache.ts_1(typ, n))
        else:
            ret = []
            for i in range(1, k):
                ret.extend(self.subs_ij(i, k - i, typ, n))

        return pack(typ, n, ret)

    def gen_one(self, k, typ):
        ret = self.gen_one_random(k, typ, 0)
        if ret is None:
            return None
        return ret[0]

    def gen_one_uf(self, uf_tree, k, typ):
        ret = self.gen_one_random_uf(uf_tree, k, typ, 0)
        if ret is None:
            return None
        return ret[0]

    def gen_one_random(self, k, typ, n):
        num = self.get_num(k, typ)
        if not num:
            return None
        return self.gen_one_raw(random.randrange(num), k, typ, n)

    def gen_one_random_uf(self, uf_tree, k, typ, n):
        num = self.get_num_uf(uf_tree, k, typ)
        if not num:
            return None
        return self.gen_one_raw_uf(uf_tree, random.randrange(num), k, typ, n)

    def gen_one_raw(self, ball, k, typ, n):
        assert k >= 1

        nf = self.normalizator(typ)
        if k == 1:
            tree, n1 = self.gen_one_leaf(ball, nf.typ_nf, n)
        else:
            tree, n1 = self.gen_one_app(ball, k, nf.typ_nf, n)

        # TODO denormalize n1 as well
        return nf.denormalize_tree(tree), n1

    def gen_one_raw_uf(self, uf_tree, ball, k, typ, n):
        assert k >= 1

        if isinstance(uf_tree, UnfinishedLeaf):
            return self.gen_one_raw(ball, k, typ, n)

        if isinstance(uf_tree, Leaf):
            assert k == 1
            assert ball == 0
            return self.gen_one_leaf_uf(uf_tree, typ, n)

        if isinstance(uf_tree, App):
            assert k > 1
            return self.gen_one_app_uf(uf_tree, ball, k, typ, n)

        assert False

    def gen_one_leaf(self, ball, typ, n):
        for res in self.cache.ts_1(typ, n):
            if not ball:
                return Leaf(res.sym, res.sub(typ)), res.n
            ball -= 1
        assert False

    def gen_one_leaf_uf(self, uf_tree, typ, n):
        ctx_declaration = self.gamma.ctx.get(uf_tree.sym, None)
        assert ctx_declaration is not None

        f = fresh(ctx_declaration.typ, typ, n)
        mu = sub.mgu(typ, f.typ)
        assert not mu.is_failed()

        return Leaf(uf_tree.sym, mu(typ)), f.n

    def gen_one_app(self, ball, k, typ, n):
        alpha, n1 = new_var(typ, n)
        typ_f = TypTerm.make_arrow(alpha, typ)

        for i in range(1, k):
            j = k - i

            for res_f in self.subs(i, typ_f, n1):
                typ_x = res_f.sub(alpha)
                for res_x in self.subs(j, typ_x, res_f.n):
                    num_fx = res_x.num * res_f.num
                    if ball < num_fx:
                        return self.gen_one_app_core(i, j,
                                                     typ,
                                                     typ_f, typ_x,
                                                     res_f, res_x)
                    ball -= num_fx
        assert False

    def gen_one_app_core(self, i, j, typ, typ_f, typ_x, res_f, res_x):
        typ_fs, deskolem_sub_f = res_f.sub(typ_f).skolemize()
        typ_xs, deskolem_sub_x = res_x.sub(typ_x).skolemize()

        s_tree_f, n = self.gen_one_random(i, typ_fs, res_x.n)
        s_tree_x, n = self.gen_one_random(j, typ_xs, n)

        assert s_tree_f is not None
        assert s_tree_x is not None

        tree_f = s_tree_f.apply_sub(sub.dot(res_x.sub, deskolem_sub_f))
        tree_x = s_tree_x.apply_sub(deskolem_sub_x)

        sigma_fx = sub.dot(res_x.sub, res_f.sub)  # not needed: .restrict(typ)
        tree_fx = App(tree_f, tree_x, sigma_fx(typ))

        return tree_fx, n

    def gen_one_app_uf(self, uf_tree, ball, k, typ, n):
        alpha, n1 = new_var(typ, n)
        typ_f = TypTerm.make_arrow(alpha, typ)
        f_uf, x_uf = uf_tree.fun, uf_tree.arg

        for i in range(1, k):
            j = k - i

            for res_f in self.subs_uf(f_uf, i, typ_f, n1):
                typ_x = res_f.sub(alpha)
                for res_x in self.subs_uf(x_uf, j, typ_x, res_f.n):
                    num_fx = res_x.num * res_f.num
                    if ball < num_fx:
                        return self.gen_one_app_core_uf(f_uf, x_uf,
                                                        i, j,
                                                        typ,
                                                        typ_f, typ_x,
                                                        res_f, res_x)
                    ball -= num_fx
        assert False

    def gen_one_app_core_uf(self, f_uf, x_uf, i, j, typ, typ_f, typ_x, res_f, res_x):
        typ_fs, deskolem_sub_f = res_f.sub(typ_f).skolemize()
        typ_xs, deskolem_sub_x = res_x.sub(typ_x).skolemize()

        s_tree_f, n = self.gen_one_random_uf(f_uf, i, typ_fs, res_x.n)
        s_tree_x, n = self.gen_one_random_uf(x_uf, j, typ_xs, n)

        assert s_tree_f is not None
        assert s_tree_x is not None

        tree_f = s_tree_f.apply_sub(sub.dot(res_x.sub, deskolem_sub_f))
        tree_x = s_tree_x.apply_sub(deskolem_sub_x)

        sigma_fx = sub.dot(res_x.sub, res_f.sub)  # not needed: .restrict(typ)
        tree_fx = App(tree_f, tree_x, sigma_fx(typ))

        return tree_fx, n

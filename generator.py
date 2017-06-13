import random
from collections import OrderedDict

import sub
from app_tree import Leaf, App, UnfinishedLeaf, split_internal_tuple
from cache import Cache
from context import Context
from normalization import Normalizator
from sub import mgu, Mover, SubRes, PreTs1Res, PreSubRes
from tracer_deco import tracer_deco
from typ import Typ, fresh, new_var, TypTerm, INTERNAL_PAIR_CONSTRUCTOR_SYM


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

    def __str__(self):
        return "Generator(...)"

    def get_num(self, k, typ):
        nf = self.normalizator(typ)
        return self.cache.get_num(k, nf.typ_nf)

    def get_num_uf(self, uf_tree, k, typ):
        return sum(sd.num for sd in self.subs_uf(uf_tree, k, typ, 0))

    def get_num_uf_smart(self, uf_tree, k):
        return sum(sd.num for sd in self.subs_uf_smart(uf_tree, k, 0))

    def ts_1_compute(self, typ, n):
        pre_ts1_res = ts1_static(self.gamma, typ, n)
        return Mover.move_pre_ts1_results(typ, n, pre_ts1_res)

    def subs_internal_pair(self, i, j, typ, n):

        i_without_cons = i - 1
        if i_without_cons == 0:
            return []

        ret = []
        typ_a, typ_b_0 = TypTerm.split_internal_pair_typ(typ)

        for res_a in self.subs(i_without_cons, typ_a, n):
            typ_b = res_a.sub(typ_b_0)
            for res_b in self.subs(j, typ_b, res_a.n):
                sigma_ab = sub.dot(res_b.sub, res_a.sub).restrict(typ)
                num_ab = res_b.num * res_a.num

                ret.append(sub.PreSubRes(num_ab, sigma_ab))

        return ret

    def subs_ij(self, i, j, typ, n):

        if TypTerm.is_internal_pair_typ(typ):
            return self.subs_internal_pair(i, j, typ, n)

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

        # @tracer_deco(log_ret=True,
        # ret_pp=lambda results: "\n".join("NUM=%d\tN=%d\n%s" % (r.num, r.n, r.sub) for r in results))

    def subs(self, k, typ, n):
        nf = self.normalizator(typ)
        results_nf = self.cache.subs(k, nf.typ_nf, n)
        ret = nf.denormalize(results_nf, n)

        return ret

    def subs_uf_smart(self, uf_tree, k, n):

        uf_leafs = uf_tree.get_unfinished_leafs()

        num_uf_leafs = len(uf_leafs)
        num_fin_leafs = uf_tree.size()

        if num_fin_leafs + num_uf_leafs > k:
            return []

        if num_uf_leafs > 0:
            hax_typ = TypTerm.make_internal_tuple([uf_leaf.typ for uf_leaf in uf_leafs])
            hax_k = k - num_fin_leafs + num_uf_leafs-1
            return self.subs(hax_k, hax_typ, n)
        elif num_fin_leafs == k:
            return [sub.PreSubRes(1, sub.Sub())]
        else:
            return []

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

    # @tracer_deco()
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

    def gen_one_uf_smart(self, uf_tree, k):

        uf_leafs = uf_tree.get_unfinished_leafs()

        num_uf_leafs = len(uf_leafs)
        num_leafs = uf_tree.size()

        if num_leafs + num_uf_leafs > k:
            return None

        # if num_uf_leafs == 1:
        #     hax_typ = uf_leafs[0].typ
        #     hax_k = k - num_leafs
        #     hax_tree = self.gen_one(hax_k, hax_typ)
        #     pass  # TODO !!!!!! special treatment maybe needed, at least for optimizations

        if num_uf_leafs > 0:
            hax_typ = TypTerm.make_internal_tuple([uf_leaf.typ for uf_leaf in uf_leafs])
            real_k = k - num_leafs
            hax_k = real_k + num_uf_leafs-1
            hax_tree = self.gen_one(hax_k, hax_typ)
            hax_subtrees = split_internal_tuple(hax_tree)
            # assert len(uf_leafs) == len(hax_subtrees)
            if len(uf_leafs) != len(hax_subtrees):
                assert False
            tree, sigma = uf_tree.replace_unfinished_leafs(hax_subtrees)

            if not tree.is_well_typed(self.gamma):
                for subtree in hax_subtrees:
                    assert subtree.is_well_typed(self.gamma)
                assert False

            return tree
        elif num_leafs == k:
            return uf_tree
        else:
            return None

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

    def gen_one_internal_pair(self, ball, k, typ, n):

        typ_a, typ_b_0 = TypTerm.split_internal_pair_typ(typ)

        for i in range(1, k):
            j = k - i

            i_without_cons = i - 1
            if i_without_cons == 0:
                continue

            for res_a in self.subs(i_without_cons, typ_a, n):
                typ_b = res_a.sub(typ_b_0)
                for res_b in self.subs(j, typ_b, res_a.n):
                    num_ab = res_b.num * res_a.num
                    if ball < num_ab:
                        return self.gen_one_internal_pair_core(i_without_cons, j, typ, typ_a, typ_b, res_a, res_b)
                    ball -= num_ab
        assert False

    def gen_one_app(self, ball, k, typ, n):

        if TypTerm.is_internal_pair_typ(typ):
            return self.gen_one_internal_pair(ball, k, typ, n)

        alpha, n1 = new_var(typ, n)
        typ_f = TypTerm.make_arrow(alpha, typ)

        for i in range(1, k):
            j = k - i
            for res_f in self.subs(i, typ_f, n1):
                typ_x = res_f.sub(alpha)
                for res_x in self.subs(j, typ_x, res_f.n):
                    num_fx = res_x.num * res_f.num
                    if ball < num_fx:
                        return self.gen_one_app_core(i, j, typ, typ_f, typ_x, res_f, res_x)
                    ball -= num_fx
        assert False

    def gen_one_internal_pair_core(self, i_without_cons, j, typ, typ_a, typ_b, res_a, res_b):
        typ_as, deskolem_sub_a = res_a.sub(typ_a).skolemize()
        typ_bs, deskolem_sub_b = res_b.sub(typ_b).skolemize()

        s_tree_a, n = self.gen_one_random(i_without_cons, typ_as, res_a.n)
        s_tree_b, n = self.gen_one_random(j, typ_bs, n)

        assert s_tree_a is not None
        assert s_tree_b is not None

        assert s_tree_a.is_well_typed(self.gamma)
        if not s_tree_b.is_well_typed(self.gamma):
            assert False

        tree_a = s_tree_a.apply_sub(sub.dot(res_b.sub, deskolem_sub_a))
        tree_b = s_tree_b.apply_sub(deskolem_sub_b)

        sigma_ab = sub.dot(res_b.sub, res_a.sub)  # not needed: .restrict(typ)
        tree_ab_typ = sigma_ab(typ)

        assert tree_ab_typ == TypTerm.make_internal_pair(tree_a.typ, tree_b.typ)

        partial_pair_typ = TypTerm.make_arrow(tree_b.typ, tree_ab_typ)
        cons_typ = TypTerm.make_arrow(tree_a.typ, partial_pair_typ)

        cons = Leaf(INTERNAL_PAIR_CONSTRUCTOR_SYM, cons_typ)
        partial_pair = App(cons, tree_a, partial_pair_typ)
        tree_ab = App(partial_pair, tree_b, tree_ab_typ)

        assert tree_a.is_well_typed(self.gamma)
        assert tree_b.is_well_typed(self.gamma)
        assert tree_ab.is_well_typed(self.gamma)

        return tree_ab, n

    def gen_one_app_core(self, i, j, typ, typ_f, typ_x, res_f, res_x):
        typ_fs, deskolem_sub_f = res_f.sub(typ_f).skolemize()
        typ_xs, deskolem_sub_x = res_x.sub(typ_x).skolemize()

        s_tree_f, n = self.gen_one_random(i, typ_fs, res_x.n)
        s_tree_x, n = self.gen_one_random(j, typ_xs, n)

        assert s_tree_f is not None
        assert s_tree_x is not None

        assert s_tree_f.is_well_typed(self.gamma)
        assert s_tree_x.is_well_typed(self.gamma)

        tree_f = s_tree_f.apply_sub(sub.dot(res_x.sub, deskolem_sub_f))
        tree_x = s_tree_x.apply_sub(deskolem_sub_x)

        sigma_fx = sub.dot(res_x.sub, res_f.sub)  # not needed: .restrict(typ)
        tree_fx = App(tree_f, tree_x, sigma_fx(typ))

        assert tree_fx.is_well_typed(self.gamma)

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

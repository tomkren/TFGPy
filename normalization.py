import sub
# from tracer_deco import tracer_deco
from collections import OrderedDict
from utils import make_enum_table
from typ import make_norm_bijection, TypVar, TypSkolem


class AbstractNormalizator:
    """
    Mandatory fields: self.typ_nf, self.n_nf
    """
    def denormalize(self, nf_sub_results):
        raise NotImplementedError

    def denormalize_tree(self, tree):
        raise NotImplementedError


def make_normalization_table(typ):

    def gather_leafs_by_type(leaf_type):
        return typ.gather_leaves(
            lambda leaf: isinstance(leaf, leaf_type),
            lambda *args: OrderedDict((a, True) for a in args)
        )

    ordered_vars = gather_leafs_by_type(TypVar)
    ordered_skolems = gather_leafs_by_type(TypSkolem)

    var_tab = make_enum_table(ordered_vars.keys(), TypVar, 0)
    make_enum_table(ordered_skolems.keys(), TypSkolem, len(var_tab), var_tab)

    return var_tab


class Normalizator(AbstractNormalizator):
    def __init__(self, typ, n):
        self.tab = make_normalization_table(typ)
        self.to_nf = sub.Sub(self.tab)
        self.typ_nf = self.to_nf(typ)
        self.n_nf = typ.get_next_var_id(n)  # we need to ensure computation will respect the *old* typ and *old* n
        # assert self.typ_nf.get_next_var_id(n) <= self.n_nf) todo neplatí pro (a->b), což je spravně

    def make_from_nf_sub(self):
        rev_tab = {v: k for k, v in self.tab.items()}
        return sub.Sub(rev_tab)

    def denormalize(self, nf_sub_results):

        from_nf = self.make_from_nf_sub()

        def denormalize_one(nf_sub_res):
            sigma_nf = nf_sub_res.sub
            sigma_table = {}
            n = self.n_nf
            for var_nf, tau_nf in sigma_nf.table.items():
                var = from_nf(var_nf)
                tau = from_nf(tau_nf)
                sigma_table[var] = tau
                n = max(n, tau.get_next_var_id())
            return sub.SubRes(nf_sub_res.num, sub.Sub(sigma_table), n)

        return [denormalize_one(nf_sub_res) for nf_sub_res in nf_sub_results]

    def denormalize_tree(self, tree):
        from_nf = self.make_from_nf_sub()
        return tree.apply_sub(from_nf)


# def make_sub_fun(table, delta):
#
#     def sub_fun(x):
#         if isinstance(x, TypVar):
#             const = TypVar
#         elif isinstance(x, TypSkolem):
#             const = TypSkolem
#         else:
#             assert False
#
#         if x in table:
#             return table[x]
#         else:
#             assert isinstance(x.name, int)
#             return const(x.name + delta)
#
#     return sub_fun


class BuggedNormalizator(AbstractNormalizator):
    def __init__(self, typ, n):
        self.typ = typ
        self.n_nf = n  # original bugged implementation (vs. typ.get_next_var_id(n))

        self.sub_to_nf, self.sub_from_nf = make_norm_bijection(typ)

        self.typ_nf = self.sub_to_nf(self.typ)
        self.tnvi = typ.get_next_var_id()

    def denormalize_tree(self, tree):
        return tree.apply_sub(self.sub_from_nf)

    def denormalize(self, nf_sub_results):
        sub_results = [self.denormalize_one(r, self.n_nf) for r in nf_sub_results]
        return sub.Mover.move_sub_results(self.typ, self.n_nf, sub_results)

    def denormalize_one(self, nf_sub_res, n):
        sigma_nf = nf_sub_res.sub
        sigma_table = {}
        nvi = max(self.tnvi, n)
        for var_nf, tau_nf in sigma_nf.table.items():
            var = self.sub_from_nf(var_nf)
            tau = self.sub_from_nf(tau_nf)

            sigma_table[var] = tau
            nvi = max(nvi, tau.get_next_var_id())

        return sub.SubRes(nf_sub_res.num, sub.Sub(sigma_table), nvi)


class NormalizatorNop(AbstractNormalizator):
    def __init__(self, typ, n):
        self.typ_nf = typ
        self.n_nf = n

    def denormalize(self, nf_sub_results):
        return [r for r in nf_sub_results]

    def denormalize_tree(self, tree):
        return tree

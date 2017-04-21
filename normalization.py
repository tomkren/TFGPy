import sub
from tracer_deco import tracer_deco
from typ import make_norm_bijection


class Normalizator:
    def __init__(self, typ):
        self.typ = typ
        self.sub_to_nf, self.sub_from_nf = make_norm_bijection(typ)

        self.typ_nf = self.sub_to_nf(self.typ)
        self.tnvi = typ.get_next_var_id()

    def denormalize(self, nf_sub_results, n):
        sub_results = [self.denormalize_one(r, n) for r in nf_sub_results]
        return sub.Mover.move_sub_results(self.typ, n, sub_results)

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


class NormalizatorNop:
    def __init__(self, typ):
        self.typ_nf = typ

    def denormalize(self, nf_sub_results, n):
        return [r for r in nf_sub_results]

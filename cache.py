from collections import namedtuple

from sub import SubRes, Mover, Ts1Res

EncodedSubData = namedtuple('EncodedSubData', ['num', 'sub_id', 'n'])
EncodedTs1Res = namedtuple('EncodedTs1Res', ['sym', 'sub_id', 'n'])


class Cache:
    def __init__(self, gen):
        self.gen = gen

        self.typ2size2data = {}
        self.typ2ts1_results = {}

        self.substitutions = []
        self.sub2id = {}

    def get_num(self, k, typ):
        size2data = self.typ2size2data.setdefault(typ, {})
        subs_data = size2data.get(k, None)
        if subs_data is None:
            subs_data = self.encode_subs(self.gen.subs_compute(k, typ, 0))
            size2data[k] = subs_data

        return sum(sd.num for sd in subs_data)

    def subs(self, k, typ, n):
        size2data = self.typ2size2data.setdefault(typ, {})
        subs_data = size2data.get(k, None)
        if subs_data is None:
            ret = self.gen.subs_compute(k, typ, 0)
            size2data[k] = self.encode_subs(ret)
        else:
            ret = self.decode_subs(subs_data)

        return Mover.move_sub_results_0(typ, n, ret)

    def ts_1(self, typ, n):
        encoded_ts1_results = self.typ2ts1_results.get(typ, None)
        if encoded_ts1_results is None:
            ret = self.gen.ts_1_compute(typ, 0)
            self.typ2ts1_results[typ] = self.encode_ts1(ret)
        else:
            ret = self.decode_ts1(encoded_ts1_results)

        return Mover.move_ts1_results_0(typ, n, ret)

    def encode_subs(self, sub_results):
        return [EncodedSubData(res.num, self.set_default_sub_id(res.sub), res.n)
                for res in sub_results]

    def decode_subs(self, subs_data):
        return [SubRes(sd.num, self.substitutions[sd.sub_id], sd.n)
                for sd in subs_data]

    def encode_ts1(self, ts1_results):
        return [EncodedTs1Res(res.sym, self.set_default_sub_id(res.sub), res.n)
                for res in ts1_results]

    def decode_ts1(self, encoded_ts1_results):
        return [Ts1Res(res.sym, self.substitutions[res.sub_id], res.n)
                for res in encoded_ts1_results]

    def set_default_sub_id(self, sub):
        sid = self.sub2id.get(sub, None)
        if sid is None:
            self.substitutions.append(sub)
            sid = len(self.substitutions) - 1
            self.sub2id[sub] = sid
        return sid

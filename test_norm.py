import unittest

from cache import CacheNop
from normalization import *
from generator import Generator
from parsers import parse_typ

# TODO totálně zemanskej shit tady byly, este zčíhnout !!!

def t1():
    return [parse_typ(t) for t in [((0, '->', (11, '->', 1)), '->', ((0, '->', 11), '->', (0, '->', 1))),
                                   ((2, '->', (1, '->', 0)), '->', ((2, '->', 1), '->', (2, '->', 0))),
                                   ((2, '->', (0, '->', 1)), '->', ((2, '->', 0), '->', (2, '->', 1))),
                                   (1, '->', (4, '->', (4, '->', (5, '->', (66, '->', (0, '->', (0, '->', (
                                       3, '->', (77, '->', (
                                           4, '->',
                                           (66, '->', (5, '->', (77, '->', (88, '->', (1, '->', 2))))))))))))))),
                                   (10, '->', (0, '->', (4, '->', (55, '->', (4, '->', (55, '->', (
                                       0, '->', (33, '->', (
                                           8, '->', (
                                           7, '->', (6, '->', (5, '->', (7, '->', (8, '->', (6, '->', 2)))))))))))))))]]


class TestNorm(unittest.TestCase):
    def test_identity(self):
        typs = t1()
        for typ in typs:
            # TODO nejak ty testy udelat lip
            n = BuggedNormalizator(typ, 0)

            self.assertEqual(typ, n.sub_from_nf(n.sub_to_nf(typ)))
            self.assertEqual(typs, [n.sub_from_nf(n.sub_to_nf(typ)) for typ in typs])


if __name__ == "__main__":
    unittest.main()

    if False:
        from test_generator import d1, d2, d3

        goal, gamma, max_k = d3()

        for k in range(3, 4):
            print("="*40)
            print(k)
            print("="*40)
            gnop = Generator(gamma, normalizator=NormalizatorNop, cache=CacheNop)
            gnf = Generator(gamma, normalizator=Normalizator, cache=CacheNop)

            print("NOP")
            ok = gnop.subs(k, goal, 0)
            print("NOPEND")
            print("NF")
            fail = gnf.subs(k, goal, 0)
            print("NFEND")
            print(len(ok))
            print(len(fail))

            if False:
                for o in ok:
                    print("-"*20)
                    print(o.num)
                    print(o.sub)

                print("@"*20)
                for o in fail:
                    print("-"*20)
                    print(o.num)
                    print(o.sub)


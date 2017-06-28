import unittest
from parsers import parse_typ
from normalization import *


def make_example_types():
    to_parse = [
        ((0, '->', (11, '->', 1)), '->', ((0, '->', 11), '->', (0, '->', 1))),
        ((TypSkolem(0), '->', (11, '->', 1)), '->', ((TypSkolem(0), '->', 11), '->', (TypSkolem(0), '->', 1))),
        ((TypSkolem(0), '->', (11, '->', 1)), '->', ((0, '->', 11), '->', (TypSkolem(0), '->', 1))),
        ((2, '->', (1, '->', 0)), '->', ((2, '->', 1), '->', (2, '->', 0))),
        ((TypSkolem(666), '->', (1, '->', 0)), '->', ((TypSkolem(555), '->', 1), '->', (TypSkolem(444), '->', 0)))
    ]
    return [parse_typ(t) for t in to_parse]


class TestNormalization(unittest.TestCase):
    def test_identity(self):
        print()
        for typ in make_example_types():
            nf = Normalizator(typ, 0)
            from_nf = nf.make_from_nf_sub()
            typ_again = from_nf(nf.to_nf(typ))
            print(typ)
            # print(typ_again)
            print(nf.typ_nf)
            print(nf.n_nf)
            self.assertEqual(typ, typ_again)


if __name__ == "__main__":
    unittest.main()

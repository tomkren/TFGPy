import unittest

from parsers import parse_typ
from typ import *


class TestTyp(unittest.TestCase):
    def test_eq(self):
        a, b = TypVar(10), TypVar(10)
        self.assertEqual(a, b)
        self.assertIsNot(a, b)

        a, b = TypVar('x1'), TypVar('x1')
        self.assertEqual(a, b)
        self.assertIsNot(a, b)

        a, b = TypVar(10), TypSymbol(10)
        self.assertNotEqual(a, b)
        self.assertIsNot(a, b)

        a, b = TypSymbol(10), TypSymbol(10)
        self.assertEqual(a, b)
        self.assertIsNot(a, b)

        a, b = TypTerm((TypVar(10), TypSymbol("S1"))), TypTerm((TypVar(10), TypSymbol("S1")))
        self.assertEqual(a, b)
        self.assertIsNot(a, b)

        a, b = TypTerm((TypVar(10), TypSymbol("S1"))), TypTerm((TypVar(11), TypSymbol("S1")))
        self.assertNotEqual(a, b)

    def test_fresh(self):
        a, b = TypTerm((TypVar(1), TypVar(1))), TypTerm((TypVar(1), TypVar(1)))
        f = fresh(a, b, 1)
        self.assertEqual(f.typ, TypTerm((TypVar(2), TypVar(2))))
        self.assertEqual(f.n, 3)

        a, b = TypTerm((TypVar(1), TypVar(2))), TypTerm((TypVar(3), TypVar(4)))
        f = fresh(a, b, 3)
        self.assertEqual(f.typ, TypTerm((TypVar(5), TypVar(6))))
        self.assertEqual(f.n, 7)

        a, b = TypTerm((TypVar(1), TypVar(2))), TypTerm((TypVar(3), TypVar(4)))
        f = fresh(a, b, 10)
        self.assertEqual(f.typ, TypTerm((TypVar(10), TypVar(11))))
        self.assertEqual(f.n, 12)

    def test_parse1(self):
        self.assertEqual(parse_typ("x"), TypVar("x"))

    def test_parse2(self):
        typ = TypTerm((
            TypVar('x1'),
            TypTerm((
                TypTerm((
                    TypSymbol('A'), TypSymbol('B')
                )),
                TypVar('x2'), TypVar('x1')
            ))
        ))

        self.assertEqual(parse_typ(('x1', (('A', 'B'), 'x2', 'x1'))), typ)
        self.assertEqual(parse_typ(['x1', [['A', 'B'], 'x2', 'x1']]), typ)
        self.assertEqual(parse_typ(['x1', [('A', 'B'), 'x2', 'x1']]), typ)

    def test_normalize(self):
        a, b = TypTerm((TypVar(666), TypVar(0))), TypTerm((TypVar(0), TypVar(1)))
        sf, st = make_var_bijection(a)
        c = a.apply_sub(sf)
        self.assertEqual(c, b)

if __name__ == "__main__":
    unittest.main()

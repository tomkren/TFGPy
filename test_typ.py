import unittest
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




if __name__ == "__main__":
    unittest.main()

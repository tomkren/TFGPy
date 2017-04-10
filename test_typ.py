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
        a, b = TypTerm((TypVar(1), TypVar(1))), TypTerm((TypVar(1), TypVar(10)))

        fr = fresh(a, b, 10)
        self.assertEqual(fr.n, 12)
        # TODO





if __name__ == "__main__":
    unittest.main()

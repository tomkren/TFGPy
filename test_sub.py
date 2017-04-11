import unittest
from sub import *
from typ import *


class TestSub(unittest.TestCase):
    def test_empty(self):
        a, b = TypVar('x1'), TypVar('x1')
        self.assertEqual(Sub(), mgu(a, b))

    def test_1(self):
        a, b = TypSymbol('S'), TypVar('x1')
        self.assertEqual(Sub({TypVar('x1'): TypSymbol('S')}), mgu(a, b))

    def test_2(self):
        a, b = TypSymbol('S'), TypSymbol('S')
        self.assertEqual(Sub(), mgu(a, b))

    def test_3(self):
        a = TypTerm((TypVar('x1'), TypVar('x1')))
        b = TypTerm((TypSymbol('A'), TypSymbol('A')))
        self.assertEqual(Sub({TypVar('x1'): TypSymbol('A')}), mgu(a, b))

    def test_4(self):
        a = TypTerm((
            TypVar('x1'),
            TypTerm((
                TypTerm((
                    TypSymbol('A'), TypSymbol('B')
                )),
                TypVar('x2'), TypVar('x1')
            ))
        ))
        b = TypTerm((
            TypTerm((
                TypSymbol('A'), TypVar('x2'),
            )),
            TypTerm((
                TypVar('x1'), TypVar('x3'),
                TypTerm((
                    TypVar('x4'), TypSymbol('B')
                ))
            ))
        ))
        u = mgu(a, b)
        self.assertEqual(Sub({
            TypVar('x1'): TypTerm((TypSymbol('A'), TypSymbol('B'))),
            TypVar('x2'): TypSymbol('B'),
            TypVar('x3'): TypSymbol('B'),
            TypVar('x4'): TypSymbol('A')
        }), u)

        # test restrict

        ru = u.restrict(a)
        self.assertEqual(ru.domain(), {TypVar('x1'), TypVar('x2')})

    def test_5(self):
        a, b = TypSymbol('A'), TypSymbol('B')

        u_sub = mgu(a, b)
        self.assertTrue(u_sub.is_failed())
        self.assertTrue(u_sub.fail_msg.startswith('Symbols mismatch'))

    def test_6(self):
        a = TypTerm((TypSymbol('A'), TypVar('x1')))
        b = TypVar('x1')

        u_sub = mgu(a, b)
        self.assertTrue(u_sub.is_failed())
        self.assertTrue(u_sub.fail_msg.startswith('Occur check'))

    def test_7(self):
        a = TypTerm((TypVar('x1'), TypVar('x1'), TypVar('x1')))
        b = TypTerm((TypSymbol('A'), TypSymbol('A')))

        u_sub = mgu(a, b)
        self.assertTrue(u_sub.is_failed())
        self.assertTrue(u_sub.fail_msg.startswith('Term length mismatch'))

    def test_8(self):
        a = TypTerm((TypVar('x1'),TypSymbol('A') ))
        b = TypSymbol('A')

        u_sub = mgu(a, b)
        self.assertTrue(u_sub.is_failed())
        self.assertTrue(u_sub.fail_msg.startswith('Kind mismatch'))

        u_sub = mgu(b, a)
        self.assertTrue(u_sub.is_failed())
        self.assertTrue(u_sub.fail_msg.startswith('Kind mismatch'))

    def test_9(self):
        a = TypVar('x1')
        b = 9

        u_sub = mgu(a, b)
        self.assertTrue(u_sub.is_failed())
        self.assertTrue(u_sub.fail_msg.startswith('Not a Typ'))

        a = 9
        b = "S"

        u_sub = mgu(a, b)
        self.assertTrue(u_sub.is_failed())
        self.assertTrue(u_sub.fail_msg.startswith('Not Typs'))

    def test_sub_call(self):

        sub = Sub({TypVar("x"): TypSymbol("S")})
        typ = TypTerm((TypVar("x"), TypVar("y")))

        self.assertEqual(sub(typ), TypTerm((TypSymbol("S"), TypVar("y"))))

    def test_parse_ctx1(self):

        sub = Sub({TypVar("x"): TypSymbol("S"), })
        self.assertEqual(sub, parse_ctx({"x": "S"}))

    def test_parse_ctx2(self):

        typ = TypTerm((
            TypTerm((
                TypSymbol('A'), TypVar('x2'),
            )),
            TypTerm((
                TypVar('x1'), TypVar('x3'),
                TypTerm((
                    TypVar('x4'), TypSymbol('B')
                ))
            ))
        ))

        sub = Sub({TypVar('x'): TypSymbol('S'), TypVar(1): typ})
        sub2 = parse_ctx({'x': 'S', 1: (['A', 'x2'], ['x1', 'x3', ('x4', 'B')])})

        self.assertEqual(sub, sub2)


if __name__ == "__main__":
    unittest.main()

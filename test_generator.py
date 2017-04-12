import unittest
from collections import OrderedDict

from parsers import parse_ctx


def make_gamma():
    return parse_ctx(OrderedDict([
        ("s", (("a", "->", ("b", "->", "c")), '->',
               (("a", "->", "b"), "->", ("a", "->", "c")))),
        ("k", ("a", "->", ("b", "->", "a"))),
        ("seri", (("Dag", 'a', 'b'), '->', (("Dag", 'b', 'c'), '->', ("Dag", 'a', 'c')))),
        ("para", (("Dag", 'a', 'b'), '->', (("Dag", 'c', 'd'), '->', ("Dag", ('P', 'a', 'c'), ('P', 'b', 'd'))))),
        ("mkDag", (("a", "->", "b"), '->', ("Dag", "a", "b"))),
        ("deDag", (("Dag", "a", "b"), '->', ("a", "->", "b"),)),
        ("mkP", ("a", "->", ("b", "->", ('P', "a", 'b')))),
        ("fst", (('P', "a", 'b'), '->', 'a')),
        ("snd", (('P', "a", 'b'), '->', 'b')),
    ]))


class TestSub(unittest.TestCase):
    pass


if __name__ == "__main__":
    #    unittest.main()

    print(str(make_gamma()))

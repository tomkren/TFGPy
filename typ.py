from collections import OrderedDict
from collections import namedtuple

import utils
import sub

FreshResult = namedtuple('FreshResult', ['typ', 'n'])


class Typ:
    def apply_mini_sub(self, key, typ):
        raise NotImplementedError

    def __eq__(self, other):
        if self is other:
            return True
        if type(other) != type(self):
            return False
        return self._eq_content(other)

    def _eq_content(self, other):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError

    def gather_leaves(self, pred, make_new):
        """
        :param make_new: constructor for a set-like structure,
                         which needs to have update method.
        """
        raise NotImplementedError

    def get_sub_keys(self):
        # TODO skolem-ready
        return self.get_vars()

    def get_vars(self):
        return self.gather_leaves(
            lambda leaf: isinstance(leaf, TypVar),
            lambda *args: set(args)
        )

    def get_next_var_id(self, acc):
        raise NotImplementedError

    def freshen_vars(self, n) -> FreshResult:
        return FreshResult(*self._freshen_vars_acc(n, {}))

    def _freshen_vars_acc(self, n, table):
        raise NotImplementedError

    def contains_var(self, var):
        raise NotImplementedError

    def apply_sub(self, sub):
        raise NotImplementedError


class TypVar(Typ):
    def __init__(self, name):
        self.name = name

    def apply_mini_sub(self, key, typ):
        if self == key:
            return typ
        return self

    def _eq_content(self, other):
        return self.name == other.name

    def contains_var(self, var):
        return self == var

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return "TypVar(%s)" % (repr(self.name))

    def gather_leaves(self, pred, make_new):
        if pred(self):
            return make_new(self)
        return make_new()

    def get_next_var_id(self, acc=0):
        if isinstance(self.name, int):
            return max(acc, self.name + 1)
        return acc

    def _freshen_vars_acc(self, n, table):
        new_var = table.get(self, None)
        if new_var is None:
            new_var = TypVar(n)
            table[self] = new_var
            n += 1
        return new_var, n

    def apply_sub(self, sub):
        if self in sub.table:
            return sub.table[self]
        return self

    def __str__(self):
        return ".%s" % self.name


class TypSymbol(Typ):
    def __init__(self, name):
        self.name = name

    def apply_mini_sub(self, *args):
        return self

    def _eq_content(self, other):
        return self.name == other.name

    def contains_var(self, var):
        return False

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return "TypSymbol(%s)" % (repr(self.name))

    def gather_leaves(self, pred, make_new):
        if pred(self):
            return make_new(self)
        return make_new()

    def get_next_var_id(self, acc=0):
        return acc

    def _freshen_vars_acc(self, n, table):
        return self, n

    def apply_sub(self, sub):
        return self

    def __str__(self):
        return str(self.name)


T_ARROW = TypSymbol('->')


class TypTerm(Typ):
    @staticmethod
    def make_arrow(left, right):
        return TypTerm((T_ARROW, left, right))

    def __init__(self, arguments):
        assert isinstance(arguments, tuple)
        assert arguments
        self.arguments = arguments

        if is_fun_type(self):
            assert len(arguments) == 3

    def apply_mini_sub(self, *args):
        return TypTerm(tuple(a.apply_mini_sub(*args) for a in self.arguments))

    def _eq_content(self, other):
        return self.arguments == other.arguments

    def contains_var(self, var):
        return any(a.contains_var(var) for a in self.arguments)

    def __hash__(self):
        return hash(self.arguments)

    def __repr__(self):
        return "TypTerm(%s)" % (repr(self.arguments))

    def gather_leaves(self, pred, make_new):
        return utils.update_union((a.gather_leaves(pred, make_new) for a in self.arguments),
                                  make_new())

    def get_next_var_id(self, acc=0):
        return max(a.get_next_var_id(acc) for a in self.arguments)

    def _freshen_vars_acc(self, n, table):
        new_arguments = []
        for a in self.arguments:
            new_term, n = a._freshen_vars_acc(n, table)
            new_arguments.append(new_term)
        return TypTerm(tuple(new_arguments)), n

    def apply_sub(self, sub):
        # TODO measure speedup
        children = tuple(a.apply_sub(sub) for a in self.arguments)
        for c, a in zip(children, self.arguments):
            if id(c) != id(a):
                return TypTerm(children)
        return self

    def __str__(self):
        if is_fun_type(self):
            op, l, r = self.arguments
            return "(%s %s %s)" % (l, op, r)

        return "(%s)" % " ".join(str(a) for a in self.arguments)


def is_fun_type(typ):
    return isinstance(typ, TypTerm) and typ.arguments[0] == T_ARROW


def split_fun_type(typ: TypTerm):
    assert is_fun_type(typ)
    return typ.arguments[1], typ.arguments[2]


def fresh(t_fresh: Typ, t_avoid: Typ, n):
    n1 = t_avoid.get_next_var_id(n)
    n2 = t_fresh.get_next_var_id(n1)

    return t_fresh.freshen_vars(n2)


def new_var(typ: Typ, n):
    n1 = typ.get_next_var_id(n)

    return TypVar(n1), n1 + 1


def make_enum_table(iterable, make_new):
    table = {}
    for num, val in enumerate(iterable):
        table[val] = make_new(num)
    return table


def make_var_bijection(typ):
    ordered_vars = typ.gather_leaves(
        lambda leaf: isinstance(leaf, TypVar),
        lambda *args: OrderedDict((a, True) for a in args)
    )
    proto_table = make_enum_table(ordered_vars.keys(), TypVar)
    table, rev_table = utils.construct_bijection(proto_table)

    return sub.Sub(table), sub.Sub(rev_table)

def get_norm_sub(typ):
    pass

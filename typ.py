from collections import namedtuple
from typing import Tuple

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

    def get_sub_keys(self):
        raise NotImplementedError

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

    def get_sub_keys(self):
        return {self}

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

    def get_sub_keys(self):
        return set()

    def get_next_var_id(self, acc=0):
        return acc

    def _freshen_vars_acc(self, n, table):
        return self, n

    def apply_sub(self, sub):
        return self


class TypTerm(Typ):
    def __init__(self, arguments):
        assert isinstance(arguments, tuple)
        assert arguments
        self.arguments = arguments

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

    def get_sub_keys(self):
        sub_keys = set()
        for a in self.arguments:
            sub_keys.update(a.get_sub_keys())
        return sub_keys

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


def fresh(t_fresh: Typ, t_avoid: Typ, n):
    n1 = t_avoid.get_next_var_id(n)
    n2 = t_fresh.get_next_var_id(n1)

    return t_fresh.freshen_vars(n2)


def new_var(typ: Typ, n):
    n1 = typ.get_next_var_id(n)

    return TypVar(n1), n1 + 1

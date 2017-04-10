class Typ:
    def apply_mini_sub(self, key, type):
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


class TypVar(Typ):
    def __init__(self, name):
        self.name = name

    def apply_mini_sub(self, key, type):
        if self == key:
            return type
        return self

    def _eq_content(self, other):
        return self.name == other.name

    def contains_var(self, var):
        return self == var

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return "TypVar(%s)"%(self.name)


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
        return "TypSymbol(%s)"%(self.name)


class TypTerm(Typ):
    def __init__(self, arguments):
        assert isinstance(arguments, tuple)
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
        return "TypTerm(%s)"%(",".join(repr(a) for a in self.arguments))

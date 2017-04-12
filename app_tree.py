class AppTree:
    pass


class App(AppTree):
    def __init__(self, fun, arg, typ):
        self.fun = fun
        self.arg = arg
        self.typ = typ

    def __repr__(self):
        return "App(%s, %s, %s)" % (repr(self.fun), repr(self.arg), repr(self.typ))

    def __str__(self):
        return "(%s %s)" % (self.fun, self.arg)


class Leaf(AppTree):
    def __init__(self, sym, typ):
        self.sym = sym
        self.typ = typ

    def __repr__(self):
        return "Leaf(%s, %s)" % (repr(self.sym), repr(self.typ))

    def __str__(self):
        return str(self.sym)

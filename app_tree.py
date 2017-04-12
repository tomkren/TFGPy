class AppTree:
    pass


class App(AppTree):
    def __init__(self, fun, arg, typ):
        self.fun = fun
        self.arg = arg
        self.typ = typ

    def __str__(self):
        return "(%s %s)"%(self.fun, self.arg)


class Leaf(AppTree):
    def __init__(self, sym, typ):
        self.sym = sym
        self.typ = typ

    def __str__(self):
        return str(self.sym)

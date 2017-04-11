class AppTree:
    pass


class App(AppTree):
    def __init__(self, fun, arg, typ):
        self.fun = fun
        self.arg = arg
        self.typ = typ


class Leaf(AppTree):
    def __init__(self, sym, typ):
        self.sym = sym
        self.typ = typ

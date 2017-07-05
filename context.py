from collections import OrderedDict
from typ import INTERNAL_PAIR_CONSTRUCTOR_SYM, TypTerm, TypVar


class ContextDeclaration:
    def __init__(self, sym, typ, is_var):
        self.sym = sym
        self.typ = typ
        self.is_var = is_var

    def __str__(self):
        return "%s : %s" % (self.sym, self.typ)


class Context:
    @staticmethod
    def from_dict(d):
        return Context.from_iterable(d.items())

    @staticmethod
    def from_iterable(iterable):
        ctx = OrderedDict()
        for k, v in iterable:
            ctx[k] = ContextDeclaration(k, v, False)
        return Context(ctx)

    def __init__(self, ctx):
        assert isinstance(ctx, OrderedDict)
        self.ctx = ctx

    def __repr__(self):
        return "Context(%s)" % (repr(self.ctx))

    def __str__(self):
        return "\n".join(str(decl) for decl in self.ctx.values())

    # TODO smazat ! nechcem přidávat
    # def add_internal_pair(self):
    #     x = TypVar('x')
    #     y = TypVar('y')
    #     xy = TypTerm.make_internal_pair(x, y)
    #
    #     sym = INTERNAL_PAIR_CONSTRUCTOR_SYM
    #     typ = TypTerm.make_arrow(x, TypTerm.make_arrow(y, xy))
    #     self.ctx[sym] = ContextDeclaration(sym, typ, False)

if __name__ == "__main__":
    from typ import TypVar

    g = Context.from_dict({
        'x': TypVar(1),
    })

    print(repr(g))

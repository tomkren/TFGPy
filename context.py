from collections import OrderedDict
from collections import namedtuple

ContextSymbol = namedtuple('ContextSymbol', ['sym', 'typ', 'is_var'])


class Context:
    @staticmethod
    def from_dict(d):
        ctx = OrderedDict()
        for k, v in d.items():
            ctx[k] = ContextSymbol(k, v, False)
        return Context(ctx)

    def __init__(self, ctx):
        assert isinstance(ctx, OrderedDict)
        self.ctx = ctx

    def __repr__(self):
        return "Context(%s)"%(repr(self.ctx))


if __name__ == "__main__":
    from typ import TypVar

    g = Context.from_dict({
        'x': TypVar(1),
    })

    print(repr(g))

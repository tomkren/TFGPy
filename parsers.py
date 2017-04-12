from collections import OrderedDict
from collections import Sequence

from context import Context
from sub import Sub
from typ import Typ, TypSymbol, TypVar, TypTerm, T_ARROW


def parse_sub(d):
    return Sub({parse_typ(k): parse_typ(v) for k, v in d.items()})


def parse_ctx(d):
    return Context.from_iterable((k, parse_typ(v)) for k, v in d.items())


def parse_typ(json):
    if isinstance(json, Typ):
        return json
    if isinstance(json, str):
        assert len(json) > 0
        return TypSymbol(json) if not json[0].islower() else TypVar(json)
    elif isinstance(json, int):
        return TypVar(json)
    elif isinstance(json, Sequence):
        args = tuple(parse_typ(x) for x in json)
        if len(args) == 3 and args[1] == T_ARROW:
            return TypTerm.make_arrow(args[0], args[2])
        return TypTerm(args)
    else:
        raise ValueError("Unsupported input value %s" % json)

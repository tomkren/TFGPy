from functools import wraps

tracer_depth = 0

DBG=False


def tracer_deco(log_ret=False, ret_pp=str):
    if not DBG:
        def deco(f):
            return f
        return deco

    def deco(f):
        @wraps(f)
        def g(*args, **kwargs):
            global tracer_depth
            print(" | " * tracer_depth, "%s(%s)" % (f.__name__, ", ".join(map(str, args[1:]))), sep='')

            tracer_depth += 1
            ret = f(*args, **kwargs)
            tracer_depth -= 1

            if log_ret:
                r = ret_pp(ret)
                for l in r.split('\n'):
                    print(" | " * tracer_depth, u" \u21B3 ", l, sep='')

            return ret

        return g

    return deco
from functools import wraps

tracer_depth = 0

enable_tracer = False


def tracer_deco(log_ret=False, ret_pp=str, print_from_arg=0, force_enable=False):
    def deco(f):
        @wraps(f)
        def g(*args, **kwargs):
            if not (enable_tracer or force_enable):
                return f(*args, **kwargs)

            global tracer_depth
            print(" | " * tracer_depth, "%s(%s, %s)" % (f.__name__,
                                                        ", ".join(map(str, args[print_from_arg:])),
                                                        ", ".join("%s=%s" % (k, v) for k, v in kwargs.items())
                                                        ), sep='')

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

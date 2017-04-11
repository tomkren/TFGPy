def union_sets(iterable):
    ret = set()
    for i in iterable:
        ret.update(i)
    return ret


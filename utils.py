import math
import sys


def update_union(iterable, acc):
    for i in iterable:
        acc.update(i)
    return acc


def construct_bijection(tab):
    table = {}
    rev_table = {}

    for source, target in tab.items():
        if source == target:
            continue
        table[source] = target
        rev_table[target] = source

    # JM: the do_todo and co_todo construction
    # might be possible to rewrite like this
    # but TODO check correctness
    # do_todo = set(rev_table.keys()) - set(table.keys())
    # co_todo = set(table.keys()) - set(rev_table.keys())

    do_todo, co_todo = [], []
    for source, target in tab.items():
        if source == target:
            continue
        if target in table:
            if source not in rev_table:
                co_todo.append(source)
        elif source in rev_table:
            if target not in table:
                do_todo.append(target)
        else:
            table[target] = source
            rev_table[source] = target

    assert len(do_todo) == len(co_todo)

    for source, target in zip(do_todo, co_todo):
        table[source] = target
        rev_table[target] = source

    return table, rev_table


def make_enum_table(iterable, make_new):
    table = {}
    for num, val in enumerate(iterable):
        table[val] = make_new(num)
    return table


def mean(l):
    if not l:
        return None
    return sum(l) / len(l)


def median(l):
    if not l:
        return None
    # make copy, s.t. we can sort in place
    l = list(l)
    l.sort()
    return l[len(l) // 2]


def sd(l):
    if not l:
        return None
    avg = mean(l)
    return math.sqrt(sum((v - avg) ** 2 for v in l))


def experiment_eval(get_one_value, repeat):
    print("experiment_eval(repeat=%d)" % repeat)
    if not repeat:
        return
    fs = []
    for i in range(repeat):
        print('.', end='', flush=True)
        fs.append(get_one_value())

    print()
    print_it_stats(fs, flush=True)
    return fs


def print_it_stats(iterator, flush=False):
    values = list(iterator)
    print(u"avg=%.3f \u00B1 %.3f\tmin=%.3f, median=%.3f, max=%.3f" % (mean(values), sd(values),
                                                                      min(values), median(values), max(values)))
    if flush:
        sys.stdout.flush()
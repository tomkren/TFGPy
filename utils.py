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

import math
import multiprocessing
import random
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


def mean_dev(l):
    if not l:
        return None
    avg = mean(l)
    return mean([abs(v - avg) for v in l])


def worker_run_with_env(fc):
    global worker_env
    return fc(worker_env)


def experiment_eval(one_iteration, make_env, repeat=10, processes=1, print_dots=False):
    if processes <= 0:
        processes = multiprocessing.cpu_count()

    print("experiment_eval(repeat=%d, processes=%d)" % (repeat, processes))
    if not repeat:
        return

    def make_worker_env():
        global worker_env
        worker_env = make_env()

    pool = multiprocessing.Pool(processes=processes, initializer=make_worker_env)

    imap = map
    if processes > 1 and repeat > 1:
        imap =pool.imap_unordered
    else:
        make_worker_env()

    scores = []
    times = []
    total_num_evals = 0
    try:
        for score, num_evals, time_spent in imap(worker_run_with_env, (one_iteration for _ in range(repeat))):
            if print_dots:
                print('.', end='', flush=True)
            scores.append(score)
            times.append(time_spent)
            total_num_evals += num_evals
    finally:
        if scores:
            print()
            print("score\t", end='')
            print_it_stats(scores)
        if times:
            print("time\t", end='')
            print_it_stats(times, flush=True)
            print("%d total evals" % total_num_evals)

    return scores


def print_it_stats(iterator, flush=False):
    values = list(iterator)
    print(u"avg=%.3f \u00B1 %.3f\tsd=%.3f min=%.3f median=%.3f max=%.3f" % (mean(values), mean_dev(values),
                                                                            sd(values),
                                                                            min(values), median(values), max(values)))
    if flush:
        sys.stdout.flush()


# Function with pretty print
class PPFunction:
    def __init__(self, raw_function, pp_name=None):
        self.raw_function = raw_function
        self.pp_name = pp_name

    def __call__(self, *args, **kwargs):
        return self.raw_function(*args, **kwargs)

    def __str__(self):
        if self.pp_name is not None:
            return self.pp_name
        return self.raw_function.__name__


def pp_function(name):
    def deco(f):
        return PPFunction(f, name)

    return deco


def sample_by_scores(choices, scores):
    assert choices
    assert len(choices) == len(scores)
    assert all(s >= 0 for s in scores)
    total = sum(scores)
    if not total:
        return random.choice(choices)
    pick = total * random.random()

    sofar = 0
    i = 0
    last = None
    while sofar < pick:
        assert i < len(choices)
        last = choices[i]
        sofar += scores[i]
        i += 1

    assert last is not None
    return last

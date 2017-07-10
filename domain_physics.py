from collections import OrderedDict
from time import time

import app_tree
import domain_koza_apptree
from fitness_cache import FitnessCache
from utils import foldr
from parsers import parse_typ, parse_ctx
from app_tree import AppTree
from typ import TypTerm
from generator import Generator

size_d = {}


def fun_typ(arg_typs, result_typ):
    def f(x, acc):
        return TypTerm.make_arrow(parse_typ(x), parse_typ(acc))

    return foldr(f, result_typ, arg_typs)


x0_sym = 'x0'
v0_sym = 'v0'
t_sym = 't'
a_sym = 'a'
half_sym = '0.5'
plus_sym = 'plus'
times_sym = 'times'
times_smart_sym = 'tms'
pair_sym = 'pair'

motion_lambda_head = (x0_sym, v0_sym, a_sym, t_sym)

P = parse_typ('P')
pair_cons_typ = fun_typ(('a', 'b'), (P, 'a', 'b'))

physics_lib_defs = {
    plus_sym: lambda x: (lambda y: x + y),
    times_sym: lambda x: (lambda y: x * y),
    times_smart_sym: lambda _: (lambda _: (lambda x: (lambda y: x * y))),
    pair_sym: lambda x: (lambda y: (x, y))
}


# print(physics_lib_defs[plus_sym](2)(8))

def get_code_str(lambda_head, tree):
    return 'lambda %s : %s' % (','.join(lambda_head), tree.eval_str())


def compile_tree(lib_defs, lambda_head, tree):
    code_str = get_code_str(lambda_head, tree)
    return eval(code_str, lib_defs)


def make_samples(begin, end, num_vals):
    delta = end - begin
    step = delta / (num_vals - 1)
    return [begin + i * step for i in range(num_vals)]


def make_tuple_samples(tuple_len, begin, end, num_vals):
    if tuple_len == 0:
        return [()]
    ret = []
    for v in make_tuple_samples(tuple_len - 1, begin, end, num_vals):
        for sample in make_samples(begin, end, num_vals):
            v_new = (*v, sample)
            ret.append(v_new)
    return ret


def motion_model(x0, v0, a, t):
    return x0 + v0 * t + 0.5 * a * t * t, v0 + a * t


def fake_model(x0, v0, a, t):
    return 23, 42


def compute_error_on_inputs(inputs, true_model, indiv_model):
    err = 0.0
    for xs in inputs:
        ys_true = true_model(*xs)
        ys_indi = indiv_model(*xs)
        for i, y_true in enumerate(ys_true):
            err += abs(y_true - ys_indi[i])
    return err


def error(lib_defs, lambda_head, tree, begin=-100, end=100, num_vals=9):
    if isinstance(tree, AppTree):
        indiv_model = compile_tree(lib_defs, lambda_head, tree)
    elif callable(tree):
        indiv_model = tree
    else:
        assert False
    inputs = make_tuple_samples(len(lambda_head), begin, end, num_vals)
    return compute_error_on_inputs(inputs, motion_model, indiv_model)


def motion_error(tree):
    return error(physics_lib_defs, motion_lambda_head, tree)


def make_simple_motion_domain(lib_defs):
    R = parse_typ('R')

    gamma = parse_ctx(OrderedDict([
        (x0_sym, R),
        (v0_sym, R),
        (t_sym, R),
        (a_sym, R),
        (half_sym, R),
        (plus_sym, fun_typ((R, R), R)),
        (times_sym, fun_typ((R, R), R)),
        (pair_sym, pair_cons_typ)
    ]))

    goal = parse_typ((P, R, R))

    title = 'Simple Motion Lib'
    optimal_size = 19

    return title, goal, gamma, optimal_size


def domain_physics(domain_maker=make_simple_motion_domain):
    title, goal, gamma, optimal_size = domain_maker(physics_lib_defs)
    cache = FitnessCache()
    inputs = make_tuple_samples(len(motion_lambda_head), -100, 100, 9)

    def fitness(individual_app_tree):
        global size_d
        size = individual_app_tree.count_nodes()[app_tree.Leaf]
        size_d[size] = size_d.get(size, 0) + 1

        s = get_code_str(motion_lambda_head, individual_app_tree)

        cres = cache.d.get(s, None)
        if cres is not None:
            return cres
        fun = eval(s, physics_lib_defs)
        err = compute_error_on_inputs(inputs, motion_model, fun)
        score = 1 / (1 + err)

        cache.update(s, score)
        return score

    return goal, gamma, fitness, (lambda: len(cache)), cache


def make_env_app_tree(smart_physics=False, **kwargs):
    raw = lambda: domain_physics()
    if smart_physics:
        raw = lambda: domain_physics(make_smart_motion_domain)

    return domain_koza_apptree.make_env_app_tree(get_raw_domain=raw, early_end_limit=1, **kwargs)


def make_smart_motion_domain(lib_defs):
    u_typ_fun_sym = parse_typ('U')

    def u(*args):
        return parse_typ((u_typ_fun_sym,) + args)

    pos_typ = u('1', '0')
    speed_typ = u('1', '-1')
    time_typ = u('0', '1')
    acceleration_typ = u('1', '-2')
    dimensionless_typ = u('0', '0')

    m, s, m1, m2, s1, s2 = map(parse_typ, ('m', 's', 'm1', 'm2', 's1', 's2'))
    u_ms = u(m, s)
    u_m1s1 = u(m1, s1)
    u_m2s2 = u(m2, s2)

    plus_m, plus_s = map(parse_typ, ('Pm', 'Ps'))

    m_vals = range(0, 2)
    s_vals = range(-2, 2)

    def make_plus_eqs(plus_pred, name_prefix, vals):
        eqs = []
        i = 1
        for a in vals:
            for b in vals:
                c = a + b
                if c in vals:
                    eq_typ = parse_typ((plus_pred, str(a), str(b), str(c)))
                    eq_sym = name_prefix + str(i)
                    eqs.append((eq_sym, eq_typ))
                    lib_defs[eq_sym] = None  # equations need no implementation
                    i += 1
        return eqs

    m_eq_typs = make_plus_eqs(plus_m, 'pm', m_vals)
    s_eq_typs = make_plus_eqs(plus_s, 'ps', s_vals)

    inputs = [
        (x0_sym, pos_typ),
        (v0_sym, speed_typ),
        (t_sym, time_typ),
        (a_sym, acceleration_typ)
    ]

    constants = [(half_sym, dimensionless_typ)]

    typ_plus = fun_typ((u_ms, u_ms), u_ms)
    typ_times = fun_typ(((plus_m, m1, m2, m), (plus_s, s1, s2, s), u_m1s1, u_m2s2), u_ms)

    operators = [
        (plus_sym, typ_plus),
        (times_smart_sym, typ_times)
    ]

    pair_stuff = [(pair_sym, pair_cons_typ)]

    gamma_list = inputs + constants + operators + m_eq_typs + s_eq_typs + pair_stuff
    gamma = parse_ctx(OrderedDict(gamma_list))

    goal = parse_typ((P, pos_typ, speed_typ))

    title = 'Smart Motion Lib'
    optimal_size = 29

    # TODO !
    return title, goal, gamma, optimal_size


def test_domain(domain_maker, lib_defs, max_k=42, show_examples=True, skip_zeros=True):
    start_time = time()

    title, goal, gamma, optimal_size = domain_maker(lib_defs)

    print('==', title, '=' * (80 - 4 - len(title)))
    print('\n in lib? | symbol : type')
    print('-' * 32)

    for sym, declaration in gamma.ctx.items():
        status = '  no'
        if sym in lib_defs:
            code = lib_defs[sym]
            if code is None:
                status = 'None'
            elif callable(code):
                status = ' Fun'
            else:
                status = ' yes'

        print(status, '\t | ', declaration)

    print('\n===> GOAL =', goal, '\n')

    gen = Generator(gamma)

    num_up_to_optimal_size = 0
    num_up_to_twice_optimal_size = 0

    print('k \t time \t num \t\t', 'example_tree' if show_examples else '')
    print('-' * (80 if show_examples else 40))
    for k in range(1, max_k + 1):
        t = time()
        num = gen.get_num(k, goal)
        dt = time() - t

        comment = ''
        if k % optimal_size == 0:
            factor = k // optimal_size
            comment = (str(factor) if factor > 1 else '') + 'OPT'

        if num > 0 or comment != '' or not skip_zeros:

            print(k, comment, '\t', ('%.2f' % dt), '\t', num, end='')

            if k <= optimal_size:
                num_up_to_optimal_size += num

            if k <= 2 * optimal_size:
                num_up_to_twice_optimal_size += num

            if num > 0 and show_examples:
                example_tree = gen.gen_one(k, goal)
                err = motion_error(example_tree)
                print('\t\t\t err=', err)  # , '\t', example_tree)
            else:
                print()

    delta_time = time() - start_time

    print()
    print('num_up_to_optimal_size :', num_up_to_optimal_size)
    print('num_up_to_twice_optimal_size :', num_up_to_twice_optimal_size)
    print()
    print('It took %.2f seconds.' % delta_time)
    print('=' * 80, '\n\n')

    return delta_time, num_up_to_optimal_size, num_up_to_twice_optimal_size


def test_domains():
    opts = {
        'lib_defs': physics_lib_defs,
        'max_k': 30,
        'show_examples': True,
        'skip_zeros': True
    }

    time_simple, ut1opt_simple, ut2opt_simple = test_domain(make_simple_motion_domain, **opts)
    time_smart, ut1opt_smart, ut2opt_smart = test_domain(make_smart_motion_domain, **opts)

    time_ratio = time_simple / time_smart
    up_to_opt_ratio = ut1opt_simple / ut1opt_smart
    up_to_2opt_ratio = ut2opt_simple / ut2opt_smart

    print('\n== RESULT STATS', '=' * 64)
    print('time_ratio       (simple/smart):\t %.4f' % time_ratio)
    print('up_to_opt_ratio  (simple/smart):\t %.1f' % up_to_opt_ratio)
    print('up_to_2opt_ratio (simple/smart):\t %.1f' % up_to_2opt_ratio)

    return time_ratio, up_to_opt_ratio, up_to_2opt_ratio


if __name__ == '__main__':
    print('fitness of motion_model:', motion_error(motion_model))
    print('fitness of fake_model:', motion_error(fake_model))
    test_domains()

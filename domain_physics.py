from collections import OrderedDict
from time import time

from utils import foldr
from parsers import parse_typ, parse_ctx
from typ import TypTerm
from generator import Generator


def U(*args):
    return parse_typ(('U',) + args)


def fun_typ(arg_typs, result_typ):
    def f(x, acc):
        return TypTerm.make_arrow(parse_typ(x), parse_typ(acc))
    return foldr(f, result_typ, arg_typs)


def make_simple_motion_domain():
    R = parse_typ('R')
    P = parse_typ('P')

    gamma = parse_ctx(OrderedDict([
        ('x0', R),
        ('v0', R),
        ('t',  R),
        ('a',  R),
        ('0.5', R),
        ('plus', fun_typ((R, R), R)),
        ('times', fun_typ((R, R), R)),
        ('pair', fun_typ(('a', 'b'), (P, 'a', 'b')))
    ]))

    goal = parse_typ((P, R, R))

    optimal_size = 19

    return goal, gamma, optimal_size


def make_motion_domain():

    m, s, m1, m2, s1, s2 = map(parse_typ, ('m', 's', 'm1', 'm2', 's1', 's2'))
    u_ms = U(m, s)
    u_m1s1 = U(m1, s1)
    u_m2s2 = U(m2, s2)

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
                    eqs.append((name_prefix+str(i), eq_typ))
                    i += 1
        return eqs

    m_eq_typs = make_plus_eqs(plus_m, 'pm', m_vals)
    s_eq_typs = make_plus_eqs(plus_s, 'ps', s_vals)

    inputs = [
        ('x0', U('1', '0')),
        ('v0', U('1', '-1')),
        ('t',  U('0', '1')),
        ('a',  U('1', '-2'))
    ]

    constants = [('0.5', U('0', '0'))]

    typ_plus = fun_typ((u_ms, u_ms), u_ms)
    typ_times = fun_typ(((plus_m, m1, m2, m), (plus_s, s1, s2, s), u_m1s1, u_m2s2), u_ms)

    operators = [
        ('plus', typ_plus),
        ('times', typ_times)
    ]

    pair_stuff = [('pair', fun_typ(('a', 'b'), ('P', 'a', 'b')))]

    gamma_list = inputs + constants + operators + m_eq_typs + s_eq_typs + pair_stuff
    gamma = parse_ctx(OrderedDict(gamma_list))

    goal = parse_typ(('P', U('1', '0'), U('1', '-1')))

    optimal_size = 29

    # TODO !
    return goal, gamma, optimal_size


def test_domain(domain_maker, max_k=42, show_examples=True, skip_zeros=True):
    start_time = time()

    goal, gamma, optimal_size = domain_maker()

    print('== gamma', '='*71)
    print(gamma)
    print()
    print('GOAL =', goal)
    print()

    gen = Generator(gamma)

    num_up_to_optimal_size = 0
    num_up_to_twice_optimal_size = 0

    print('k \t num \t\t', 'example_tree' if show_examples else '')
    print('-'*(80 if show_examples else 40))
    for k in range(1, max_k+1):
        num = gen.get_num(k, goal)
        if num > 0 or not skip_zeros:
            print(k, ('OPT' if k == optimal_size else ''), '\t', num, end='')

            if k <= optimal_size:
                num_up_to_optimal_size += num

            if k <= 2*optimal_size:
                num_up_to_twice_optimal_size += num

            if show_examples:
                example_tree = gen.gen_one(k, goal)
                print('\t\t\t', example_tree)
            else:
                print()

    print()
    print('num_up_to_optimal_size :', num_up_to_optimal_size)
    print('num_up_to_twice_optimal_size :', num_up_to_twice_optimal_size)
    print()
    print('It took', time() - start_time, 'seconds.')
    print('='*80, '\n\n')


def test_domains():
    opts = {'max_k': 60, 'show_examples': False, 'skip_zeros': True}
    test_domain(make_simple_motion_domain, **opts)
    test_domain(make_motion_domain, **opts)


if __name__ == '__main__':
    test_domains()

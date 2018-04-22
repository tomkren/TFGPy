from collections import OrderedDict
from time import time
import os
import json
from PIL import Image, ImageDraw, ImageMath, ImageStat  # Using https://pillow.readthedocs.io
import imagehash  # Using https://pypi.python.org/pypi/ImageHash
import numpy as np

from parsers import parse_typ, parse_ctx, fun_typ
from generator import Generator
from generator_static import ts


def main():

    save_hand_crafted_examples((512, 512))  # Run quick example generation ..

    path = 'imgs/gen'
    dim_size = 32
    img_size = (dim_size, dim_size)

    hash_opts = {
        'hash_size': 32,
        'highfreq_factor': 4
    }

    gen_opts_full = {
        'max_tree_size': 51,
        'exhaustive_generating_limit': 250000,
        'sample_method': {
            'name': 'fixed_attempts',
            'num_attempts': 20000
        },
        'domain_maker': 'family_1',
        'hash_opts': hash_opts,
        'img_size': img_size,
        'path': path
    }

    gen_opts_requested = {
        'max_tree_size': 13,
        'exhaustive_generating_limit': 250000,
        'sample_method': {
            'name': 'fixed_attempts',
            'num_attempts': 100000
        },
        'domain_maker': 'family_1',
        'hash_opts': hash_opts,
        'img_size': (128, 128),
        'path': path
    }

    gen_opts_test = {
        'max_tree_size': 17,
        'exhaustive_generating_limit': 2500,
        'sample_method': {
            'name': 'fixed_attempts',
            'num_attempts': 100
        },
        'domain_maker': 'family_1',
        'hash_opts': hash_opts,
        'img_size': img_size,
        'path': path
    }

    # generate_dataset(gen_opts_test)
    # generate_dataset(gen_opts_full)
    generate_dataset(gen_opts_requested)


def main_process_results():

    dataset_id = '003'

    results_dir_path = 'imgs/results/results_' + dataset_id + '/'

    inputs_path = results_dir_path + 'dev_imgs.txt'
    outputs_path = results_dir_path + 'prefixes_out.txt'
    report_path = results_dir_path + 'report.html'

    in_imgs_path = results_dir_path + 'imgs/'
    out_imgs_path = results_dir_path + 'imgs_out/'

    ensure_dir(out_imgs_path)
    open(report_path, 'w').close()

    worst_err = 0.0
    sum_err = 0.0

    # report = ''

    i = 0
    rows = []
    num_absolute_matches = 0

    with open(inputs_path) as f_in:
        with open(outputs_path) as f_out:

            # report += '<table border="1">\n'
            # report += '<tr><th>input</th><th>output</th><th>error</th>\n'

            while True:

                in_line = f_in.readline().strip()
                out_line = f_out.readline().strip()

                if (not in_line) or (not out_line):
                    break

                corrected_code = from_prefix_notation_family_1(out_line)

                in_img_path = in_imgs_path + in_line
                out_img_path = out_imgs_path + in_line

                input_im = Image.open(in_img_path)

                if not os.path.isfile(out_imgs_path):
                    render_to_file(out_img_path, input_im.size, corrected_code)

                output_im = Image.open(out_img_path)

                err = imgs_err(input_im, output_im)

                sum_err += err
                if err == 0.0:
                    num_absolute_matches += 1
                if err > worst_err:
                    worst_err = err

                in_img_html = '<img src="imgs/%s">' % in_line
                out_img_html = '<img src="imgs_out/%s">' % in_line

                rows.append((in_img_html, out_img_html, err))

                # report += '<tr><td>%s</td><td>%s</td><td><pre>%.10f</pre></td></tr>\n'%(in_img_html,out_img_html,err)
                print("%s -> %s ... %.10f" % (in_line, out_line, err))
                i += 1

            # report += '</table>\n'

    stats = '\n'
    stats += 'STATS:\n'
    stats += 'Number of test instances: %d\n' % i
    stats += 'Number of absolute matches: %d\n' % num_absolute_matches
    stats += 'Percent absolute matches: %f\n' % (100.0 * num_absolute_matches / i)
    stats += 'Average error: %.5f\n' % (sum_err / i)
    stats += 'Worst error: %.10f\n' % worst_err

    print(stats)
    print('Generating report.html ...')

    table = '<table border="1">\n'
    table += '<tr><th>in</th><th>out</th><th>error</th>\n'
    for row in sorted(rows, key=lambda r: -r[2]):
        table += '<tr><td>%s</td><td>%s</td><td><pre>%.10f</pre></td></tr>\n' % row
    table += '</table>\n'

    report_stats = '<pre>%s</pre>' % stats

    with open(report_path, 'w') as f_report:
        f_report.write(report_stats + '<br><br>\n' + table)

    print('Done.')


def generate_dataset(gen_opts):
    start_time = time()

    gen_opts['paths'] = init_files(gen_opts['path'])
    save_stats_header(gen_opts)

    domain_maker = family_lib[gen_opts['domain_maker']]
    goal, gamma = domain_maker()
    gen = Generator(gamma)

    img_hashes = {}
    next_img_id = 1
    attempt = 0

    max_tree_size = gen_opts['max_tree_size']
    exhaustive_generating_limit = gen_opts['exhaustive_generating_limit']
    sample_method = gen_opts['sample_method']
    sample_method_name = sample_method['name']

    for tree_size in range(1, max_tree_size + 1):
        one_size_start_time = time()

        num_trees = gen.get_num(tree_size, goal)

        next_img_id_start = next_img_id

        print('tree_size =', tree_size, "-> num_trees =", num_trees)

        if num_trees > 0:

            if num_trees < exhaustive_generating_limit:

                gen_method_name = 'exhaustive'
                num_attempts = num_trees

                for tree_data in ts(gamma, tree_size, goal, 0):

                    tree = tree_data.tree
                    attempt += 1

                    next_img_id = generate_step(gen_opts, tree, img_hashes, tree_size, next_img_id, attempt)

            else:

                gen_method_name = sample_method_name

                if sample_method_name == 'fixed_attempts':

                    num_attempts = sample_method['num_attempts']

                    for i_sample in range(num_attempts):

                        tree = gen.gen_one(tree_size, goal)
                        attempt += 1

                        next_img_id = generate_step(gen_opts, tree, img_hashes, tree_size, next_img_id, attempt)

                else:
                    num_attempts = -1
                    print('WARNING: Using unsupported sampling method.')

            new_for_this_size = next_img_id - next_img_id_start
            one_size_delta_time = time() - one_size_start_time

            save_stats_size_info(gen_opts, tree_size, num_trees, gen_method_name, num_attempts, new_for_this_size, one_size_delta_time)

    # save stats and we are done ..
    num_generated_trees = next_img_id - 1
    delta_time = time() - start_time
    save_stats_footer(gen_opts, num_generated_trees, attempt, delta_time)
    print(gen_opts['stats'])


def generate_step(gen_opts, tree, img_hashes, tree_size, next_img_id, attempt):
    img_code = tree.to_sexpr_json()
    im, img_hash = render_to_img_with_phash(gen_opts, img_code)
    if img_hash not in img_hashes:
        img_hashes[img_hash] = tree_size
        save_generated_tree_data(gen_opts, next_img_id, im, img_code, tree, tree_size, attempt)
        return next_img_id + 1
    else:
        return next_img_id


def init_files(path):

    if not path.endswith('/'):
        path += '/'

    imgs_path = path + 'imgs/'
    ensure_dir(imgs_path)

    paths = {
        'img_pattern': imgs_path + '%08d.png',
        'stats': path + 'stats.md',
        'jsons': path + 'jsons.txt',
        'prefix': path + 'prefix.txt',
        'roots': path + 'roots.txt'
    }

    open(paths['stats'], 'w').close()
    open(paths['jsons'], 'w').close()
    open(paths['prefix'], 'w').close()
    open(paths['roots'], 'w').close()

    return paths


def save_generated_tree_data(gen_opts, img_id, im, img_code, tree, tree_size, attempt):
    paths = gen_opts['paths']

    im.save(paths['img_pattern'] % img_id, 'PNG')

    root_sym = root_symbol(img_code)
    prefix_code = to_prefix_notation(img_code)

    append_line(paths['roots'], root_sym)
    append_line(paths['prefix'], prefix_code)
    append_line(paths['jsons'], str(img_code))

    print('%-7d->' % img_id, paths['img_pattern'] % img_id, "attempt=%d" % attempt, "tree_size=%d" % tree_size)
    # print('\t\ttree =', tree)
    # print('\t\ts-expr =', img_code)
    print('\t\ttree =', prefix_code)


def save_stats_header(gen_opts):
    stats = '# Stats #\n\n'
    stats += '## gen_opts ##\n\n'
    gen_opts_pretty_json = json.dumps(gen_opts, sort_keys=True, indent=2, separators=(',', ': '))
    stats += '```json\n%s\n```\n\n' % gen_opts_pretty_json
    stats += '## Stats for tree sizes ##\n\n'
    row = 'Tree size', 'Num of all trees', 'Generating method', 'Attempts', 'New trees', 'New/Attempts %', 'Time'
    stats += '| %-9s | %-40s | %-17s | %-10s | %-10s | %-14s | %-14s |\n' % row
    stats += '| %s | %s | %s | %s | %s | %s | %s |\n' % ('-'*9, '-'*40, '-'*17, '-'*10, '-'*10, '-'*14, '-'*14)
    gen_opts['stats'] = ''
    append_stats(gen_opts, stats)


def save_stats_size_info(gen_opts, tree_size, num_trees, gen_method_name, num_attempts, new_for_this_size, time):
    new_to_attempts_percent = (100.0 * new_for_this_size) / num_attempts
    row = tree_size, num_trees, gen_method_name, num_attempts, new_for_this_size, new_to_attempts_percent, time
    stats = '| %-9d | %-40d | %-17s | %-10d | %-10d | %-14.2f | %-14.2f |\n' % row
    append_stats(gen_opts, stats)


def save_stats_footer(gen_opts, num_generated_trees, attempts, delta_time):
    stats = '\n## Final stats ##\n\n'
    stats += '* Num Generated Images: %d\n' % num_generated_trees
    stats += '* Num Attempts: %d\n' % attempts
    stats += '* Generating Time: %.2f s\n' % delta_time
    stats += '* Average attempt time: %f s\n' % (delta_time / attempts)
    stats += '* Average new tree time: %f s\n' % (delta_time / num_generated_trees)

    append_stats(gen_opts, stats)


def append_stats(gen_opts, stats):
    with open(gen_opts['paths']['stats'], 'a') as stats_file:
        stats_file.write(stats)
    gen_opts['stats'] += stats


def make_family_1():
    t_img = parse_typ('I')  # simple Image type

    t_op2 = fun_typ((t_img, t_img), t_img)  # Simple binary operation
    t_op4 = fun_typ((t_img, t_img, t_img, t_img), t_img)  # Simple tetra operation

    goal = t_img
    gamma = parse_ctx(OrderedDict([
        (H, t_op2),
        (V, t_op2),
        (Q, t_op4),
        (W, t_img),
        (B, t_img)
    ]))

    return goal, gamma


family_lib = {
    'family_1': make_family_1
}

H, V, Q, C = 'h', 'v', 'q', 'c'

h = lambda c1, c2: [H, c1, c2]
v = lambda c1, c2: [V, c1, c2]
q = lambda c1, c2, c3, c4: [Q, c1, c2, c3, c4]
c = lambda r, g, b: [C, r, g, b]

W, B, G = 'W', 'B', 'G'
Re, Gr, Bl = 'r', 'g', 'b'

Colors = {
    W: (255, 255, 255), B: (0, 0, 0), G: (128, 128, 128),
    Re: (255, 0, 0), Gr: (0, 255, 0), Bl: (0, 0, 255)
}

# TODO: generate automatically !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
arity_dict_family_1 = {
    H: 2,
    V: 2,
    Q: 4,
    W: 0,
    B: 0,

    C: 3,
    G: 0,
    Re: 0,
    Gr: 0,
    Bl: 0
}


def make_hand_crafted_examples():
    code_001 = h(v(G, h(v(G, h(v(G, h(v(G, B), B)), B)), B)), B)
    code_002 = q(Re, Gr, Bl, q(Re, Gr, Bl, q(Re, Gr, Bl, q(Re, Gr, Bl, q(Re, Gr, Bl, q(Re, Gr, Bl, q(Re, Gr, Bl, q(Re, Gr, Bl, G))))))))
    elephant = q(G, q(q(W, W, q(G, G, B, G), W), W, G, W), q(W, G, q(G, G, G, q(W, W, G, W)), q(G, G, W, G)),
                q(q(q(W, q(W, W, G, q(G, W, q(q(W, W, W, G), G, W, W), W)), W, G), G, W, W), W, q(W, W, q(W, W, G, W), W), W))

    simpler_elephant = q(G, q(W, W, G, W), q(W, G, q(G, G, G, W), q(G, G, W, G)), q(q(q(W, q(W, W, G, q(G, W, q(q(W, W, W, G), G, W, W), W)), W, G), G, W, W), W, q(W, W, W, W), W))

    return [code_001, code_002, elephant, simpler_elephant]


def save_hand_crafted_examples(img_size):
    codes = make_hand_crafted_examples()

    dir_path = 'imgs/handmade/'
    ensure_dir(dir_path)
    filename_pat = dir_path + '%03d.png'

    for i, code in enumerate(codes):
        render_to_file(filename_pat % (i+1), img_size, code)


def append_line(filename, line):
    with open(filename, 'a') as f:
        f.write("%s\n" % line)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def root_symbol(code):
    if isinstance(code, list):
        return root_symbol(code[0])
    else:
        return code


def to_prefix_notation(code):
    return ' '.join(to_prefix_json(code))


def to_prefix_json(code):
    if isinstance(code, list):
        return sum([to_prefix_json(arg) for arg in code], [])
    else:
        return [code]


def from_prefix_notation(prefix_notation_str, arity_dict, default_sym):
    prefix_list = prefix_notation_str.strip().split()
    mismatch = compute_prefix_mismatch(prefix_list, arity_dict)

    if mismatch < 0:
        del prefix_list[mismatch:]  # Too many symbols, we cut off last -mismatch elements.
    elif mismatch > 0:
        prefix_list += [default_sym] * mismatch  # To few symbols, we add default symbols.

    return from_prefix_list(prefix_list, arity_dict)


def from_prefix_list(prefix_list, arity_dict):
    stack = []
    for sym in reversed(prefix_list):
        arity = arity_dict[sym]
        if arity == 0:
            stack.append(sym)
        else:
            code = [sym]
            for i in range(arity):
                code.append(stack.pop())
            stack.append(code)
    return stack.pop()


def compute_prefix_mismatch(prefix_list, arity_dict):

    mismatch = 1  # Initially, we have one "open node" in the root.

    for sym in prefix_list:

        if sym not in arity_dict:
            raise ValueError("Unsupported symbol '%s' in '%s'." % (sym, prefix_list))

        arity = arity_dict[sym]
        mismatch += arity - 1

    return mismatch


def from_prefix_notation_family_1(prefix_notation_str):
    return from_prefix_notation(prefix_notation_str, arity_dict_family_1, W)


def test_img_code(img_code):
    prefix_code = to_prefix_notation(img_code)
    test_code = from_prefix_notation_family_1(prefix_code)
    if str(test_code) == str(img_code):
        print('OK: %s' % test_code)
    else:
        raise ValueError('TEST FAILED!')


def main_test_sort():

    for row in sorted([('abc', 12.1), ('cde', 120.1), ('efg', 1.21)], key=lambda r: -r[1]):
        print(row)


def main_test():

    print(from_prefix_notation_family_1("W"))
    print(from_prefix_notation_family_1("B"))
    print(from_prefix_notation_family_1("h"))
    print(from_prefix_notation_family_1("h v B h B"))
    print(from_prefix_notation_family_1("h v B h B W W W B B B"))

    examples = make_hand_crafted_examples()

    for code in examples:
        test_img_code(code)

    img_size = (256, 256)
    elephant = examples[2]
    simpler_elephant = examples[3]

    im1 = render_to_img(img_size, elephant)
    im2 = render_to_img(img_size, simpler_elephant)

    i1 = np.array(im1).astype(float)
    i2 = np.array(im2).astype(float)

    diff = np.abs(i1 - i2)
    err = np.sum(diff) / (3 * img_size[0] * img_size[1] * 255)

    print(float(err))

    im1.show()
    im2.show()

    diff = ImageMath.eval("abs(a - b)", a=im1.convert('L'), b=im2.convert('L'))
    diff.show()


def robust_err(input_img_path, output_prefix_notation_str):
    im1 = Image.open(input_img_path)
    corrected_code = from_prefix_notation_family_1(output_prefix_notation_str)
    im2 = render_to_img(im1.size, corrected_code)
    return imgs_err(im1, im2)


def codes_err(code1, code2, img_size):
    im1 = render_to_img(img_size, code1)
    im2 = render_to_img(img_size, code2)
    return imgs_err(im1, im2)


def imgs_err(im1, im2):
    if im1.size != im2.size:
        raise ValueError("Images must have the same size.")

    i1 = np.array(im1).astype(float)
    i2 = np.array(im2).astype(float)

    diff = np.abs(i1 - i2)
    err = np.sum(diff) / (3 * im1.size[0] * im1.size[1] * 255)
    return float(err)


def render(code, zoom, draw):

    if isinstance(code, list):
        func_name = code[0]

        if is_split_operator(func_name):

            new_zooms = split_zoom(func_name, zoom)

            if len(new_zooms) != len(code) - 1:
                raise ValueError('Split operator mus have 2 args.')

            for i in range(0, len(new_zooms)):
                render(code[i+1], new_zooms[i], draw)

        elif is_color_encoding(func_name):

            if len(code) != 4:
                raise ValueError('Color encoding must have 3 args.')

            render_color((code[1], code[2], code[3]), zoom, draw)

        else:
            raise ValueError('Unsupported function', func_name)

    elif isinstance(code, str):
        render_color(decode_color(code), zoom, draw)

    else:
        raise ValueError("Unsupported code format.")


def render_color(color, zoom, draw):
    draw.rectangle(zoom, fill=color)


def is_split_operator(func_name):
    return func_name == H or func_name == V or func_name == Q


def is_color_encoding(func_name):
    return func_name == C


def split_zoom(func_name, zoom):
    x1, y1, x2, y2 = zoom
    if func_name == H:
        y_split = round((y1 + y2) / 2)
        return [(x1, y1, x2, y_split), (x1, y_split, x2, y2)]
    elif func_name == V:
        x_split = round((x1 + x2) / 2)
        return [(x1, y1, x_split, y2), (x_split, y1, x2, y2)]
    elif func_name == Q:
        x_split = round((x1 + x2) / 2)
        y_split = round((y1 + y2) / 2)
        return [
            (x1, y1, x_split, y_split), (x_split, y1, x2, y_split),
            (x1, y_split, x_split, y2), (x_split, y_split, x2, y2)
        ]
    else:
        raise ValueError("Unsupported function", func_name)


def decode_color(color_code):
    color = Colors.get(color_code, None)
    if color is None:
        raise ValueError('Unsupported color code', color_code)
    return color


def render_to_img(img_size, img_code):
    im = Image.new('RGB', img_size)

    zoom = (0, 0, im.size[0], im.size[1])
    draw = ImageDraw.Draw(im)

    render(img_code, zoom, draw)

    del draw
    return im


def render_to_img_with_phash(gen_opts, img_code):
    im = render_to_img(gen_opts['img_size'], img_code)

    hash_opts = gen_opts['hash_opts']
    img_hash = imagehash.phash(im, hash_opts['hash_size'], hash_opts['highfreq_factor'])
    # print('\t img_hash =', img_hash)

    return im, img_hash


def render_to_file(filename, img_size, img_code):
    im = render_to_img(img_size, img_code)
    im.save(filename, 'PNG')


if __name__ == '__main__':
    # main()
    # main_test()
    main_process_results()

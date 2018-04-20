from collections import OrderedDict
from time import time
import os
import json
from PIL import Image, ImageDraw  # Using https://pillow.readthedocs.io
import imagehash  # Using https://pypi.python.org/pypi/ImageHash

from parsers import parse_typ, parse_ctx, fun_typ
from generator import Generator
from generator_static import ts


def main():

    make_hand_crafted_examples((512, 512))  # Run quick example generation ..

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

    generate_dataset(gen_opts_test)
    # generate_dataset(gen_opts_full)


def generate_dataset(gen_opts):
    start_time = time()

    gen_opts['paths'] = init_files(gen_opts['path'])
    save_stats_header(gen_opts)

    domain_maker = family_lib[gen_opts['domain_maker']]
    goal, gamma = domain_maker()
    gen = Generator(gamma)

    img_hashes = {}
    next_img_id = 1

    max_tree_size = gen_opts['max_tree_size']
    exhaustive_generating_limit = gen_opts['exhaustive_generating_limit']
    sample_method = gen_opts['sample_method']
    sample_method_name = sample_method['name']

    for tree_size in range(1, max_tree_size + 1):
        num_trees = gen.get_num(tree_size, goal)

        next_img_id_start = next_img_id

        print('tree_size =', tree_size, "-> num_trees =", num_trees)

        if num_trees > 0:

            if num_trees < exhaustive_generating_limit:

                gen_method_name = 'exhaustive'
                num_attempts = num_trees

                for tree_data in ts(gamma, tree_size, goal, 0):

                    tree = tree_data.tree
                    next_img_id = generate_step(gen_opts, tree, img_hashes, tree_size, next_img_id)

            else:

                gen_method_name = sample_method_name

                if sample_method_name == 'fixed_attempts':

                    num_attempts = sample_method['num_attempts']

                    for i_sample in range(num_attempts):

                        tree = gen.gen_one(tree_size, goal)
                        next_img_id = generate_step(gen_opts, tree, img_hashes, tree_size, next_img_id)

                else:
                    num_attempts = -1
                    print('WARNING: Using unsupported sampling method.')

            new_for_this_size = next_img_id - next_img_id_start

            save_stats_size_info(gen_opts, tree_size, num_trees, gen_method_name, num_attempts, new_for_this_size)

    # save stats and we are done ..
    num_generated_trees = next_img_id - 1
    delta_time = time() - start_time
    save_stats_footer(gen_opts, num_generated_trees, delta_time)
    print(gen_opts['stats'])


def generate_step(gen_opts, tree, img_hashes, tree_size, next_img_id):
    img_code = tree.to_sexpr_json()
    im, img_hash = render_to_img_with_phash(gen_opts, img_code)
    if img_hash not in img_hashes:
        img_hashes[img_hash] = tree_size
        save_generated_tree_data(gen_opts, next_img_id, im, img_code, tree, tree_size)
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


def save_generated_tree_data(gen_opts, img_id, im, img_code, tree, tree_size):
    paths = gen_opts['paths']

    im.save(paths['img_pattern'] % img_id, 'PNG')

    root_sym = root_symbol(img_code)
    prefix_code = to_prefix_notation(img_code)

    append_line(paths['roots'], root_sym)
    append_line(paths['prefix'], prefix_code)
    append_line(paths['jsons'], str(img_code))

    print('%-7d->' % img_id, paths['img_pattern'] % img_id, "tree_size=%d" % tree_size)
    print('\t\ttree =', tree)
    print('\t\ts-expr =', img_code)
    print('\t\tprefix =', prefix_code)


def save_stats_header(gen_opts):
    stats = '# Stats #\n\n'
    stats += '## gen_opts ##\n\n'
    gen_opts_pretty_json = json.dumps(gen_opts, sort_keys=True, indent=2, separators=(',', ': '))
    stats += '```json\n%s\n```\n\n' % gen_opts_pretty_json
    stats += '## Stats for tree sizes ##\n\n'
    row = 'Tree size', 'Num of all trees', 'Generating method', 'Attempts', 'New trees', 'New/Attempts %'
    stats += '| %-9s | %-40s | %-17s | %-10s | %-10s | %-14s |\n' % row
    stats += '| %s | %s | %s | %s | %s | %s |\n' % ('-'*9, '-'*40, '-'*17, '-'*10, '-'*10, '-'*14)
    gen_opts['stats'] = ''
    append_stats(gen_opts, stats)


def save_stats_size_info(gen_opts, tree_size, num_trees, gen_method_name, num_attempts, new_for_this_size):
    new_to_attempts_percent = (100.0 * new_for_this_size) / num_attempts
    row = tree_size, num_trees, gen_method_name, num_attempts, new_for_this_size, new_to_attempts_percent
    stats = '| %-9d | %-40d | %-17s | %-10d | %-10d | %-14.2f |\n' % row
    append_stats(gen_opts, stats)


def save_stats_footer(gen_opts, num_generated_trees, delta_time):
    stats = '\n## Final stats ##\n\n'
    stats += '* Num Generated Images: %d\n' % num_generated_trees
    stats += '* Generating Time: %.2f s\n' % delta_time
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


def make_hand_crafted_examples(img_size):
    code_001 = h(v(G, h(v(G, h(v(G, h(v(G, B), B)), B)), B)), B)
    code_002 = q(Re, Gr, Bl, q(Re, Gr, Bl, q(Re, Gr, Bl, q(Re, Gr, Bl, q(Re, Gr, Bl, q(Re, Gr, Bl, q(Re, Gr, Bl, q(Re, Gr, Bl, G))))))))
    elephant = q(G, q(q(W, W, q(G, G, B, G), W), W, G, W), q(W, G, q(G, G, G, q(W, W, G, W)), q(G, G, W, G)),
                q(q(q(W, q(W, W, G, q(G, W, q(q(W, W, W, G), G, W, W), W)), W, G), G, W, W), W, q(W, W, q(W, W, G, W), W), W))

    codes = [code_001, code_002, elephant]

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
    main()

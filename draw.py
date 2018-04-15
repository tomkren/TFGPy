from PIL import Image, ImageDraw  # Using https://pillow.readthedocs.io
from collections import OrderedDict
from time import time
import os
import imagehash  # Using https://pypi.python.org/pypi/ImageHash

from parsers import parse_typ, parse_ctx, fun_typ
from generator import Generator
from generator_static import ts


def main():

    make_hand_crafted_examples((512, 512))  # Run quick example generation ..

    path = 'imgs/gen'
    dim_size = 32
    img_size = (dim_size, dim_size)
    hash_size = 32
    highfreq_factor = 4

    gen_opts = {
        'exhaustive_generating_limit': 10000,  # 250000,
        'sample_method': {
            'name': 'simple_sampling',
            'num_attempts': 100000
        }
    }

    domain_maker = make_family_1
    max_tree_size = 13  # 7 -> ~264, 9 -> ~2602, 11 -> ~28148

    generate_dataset(path, domain_maker, max_tree_size, gen_opts, img_size, hash_size, highfreq_factor)


def generate_dataset(path, domain_maker, max_tree_size, gen_opts, img_size, hash_size=8, highfreq_factor=4):
    start_time = time()

    exhaustive_generating_limit = gen_opts['exhaustive_generating_limit']

    if not path.endswith('/'):
        path += '/'

    imgs_path = path + 'imgs/'

    ensure_dir(imgs_path)

    # img_filename_pat = imgs_path + '%07d.png'
    stats_filename = path + 'stats.txt'

    # jsons_filename = path + 'jsons.txt'
    # prefix_filename = path + 'prefix.txt'
    # roots_filename = path + 'roots.txt'

    paths = {
        'img_pattern': imgs_path + '%08d.png',
        'jsons': path + 'jsons.txt',
        'prefix': path + 'prefix.txt',
        'roots': path + 'roots.txt'
    }

    open(paths['jsons'], 'w').close()
    open(paths['prefix'], 'w').close()
    open(paths['roots'], 'w').close()

    goal, gamma = domain_maker()
    gen = Generator(gamma)

    img_hashes = {}
    stats_for_sizes = []

    i = 1

    for k in range(1, max_tree_size + 1):
        num = gen.get_num(k, goal)

        print('size =', k, "-> num =", num)

        if num > 0:

            new_for_this_size = 0

            if num < exhaustive_generating_limit:

                for tree_data in ts(gamma, k, goal, 0):

                    tree = tree_data.tree

                    img_code = tree.to_sexpr_json()

                    im = render_to_img(img_size, img_code)
                    img_hash = imagehash.phash(im, hash_size, highfreq_factor)
                    # print('\t img_hash =', img_hash)

                    if img_hash not in img_hashes:
                        img_hashes[img_hash] = k   # img_code

                        save_generated_tree_data(i, im, img_code, tree, paths)

                        # im.save(img_filename_pat % i, 'PNG')

                        # root_sym = root_symbol(img_code)
                        # prefix_code = to_prefix_notation(img_code)

                        # append_line(roots_filename, root_sym)
                        # append_line(prefix_filename, prefix_code)
                        # append_line(jsons_filename, str(img_code))

                        # print('\t', i)
                        # print('\t\t example_tree   =', tree)
                        # print('\t\t s-expr =', img_code)
                        # print('\t\t prefix =', prefix_code)
                        # print('\t\t img_hash =', img_hash)

                        i += 1
                        new_for_this_size += 1
            else:

                # example_tree = gen.gen_one(k, goal)

                pass

            new_to_all_percent = (100.0 * new_for_this_size) / num
            stats_for_sizes.append((k, num, new_for_this_size, new_to_all_percent))

    num_generated_trees = i - 1
    delta_time = time() - start_time

    stats = 'Num Generated Images: %d\n' % num_generated_trees
    stats += 'Generating Time: %.2f s\n' % delta_time
    stats += 'Image size: %d×%d\n' % img_size
    stats += 'pHash params (size, highfreq_factor): %d, %d\n' % (hash_size, highfreq_factor)

    stats += 'Stats for Sizes:\n'
    stats += '%-15s%-20s%-20s%s\n' % ('Tree size', 'Num of all trees', 'New trees', 'New/All %')
    for o in stats_for_sizes:
        stats += '%-15d%-20d%-20d%.2f\n' % o

    with open(stats_filename, 'w') as stats_file:
        stats_file.write(stats)

    print('\n'+stats)


def save_generated_tree_data(i, im, img_code, tree, paths):

    im.save(paths['img_pattern'] % i, 'PNG')

    root_sym = root_symbol(img_code)
    prefix_code = to_prefix_notation(img_code)

    append_line(paths['roots'], root_sym)
    append_line(paths['prefix'], prefix_code)
    append_line(paths['jsons'], str(img_code))

    print('%-8d->' % i, paths['img_pattern'] % i)
    print('\t\ttree =', tree)
    print('\t\ts-expr =', img_code)
    print('\t\tprefix =', prefix_code)


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


def render_to_img(img_size, code):
    im = Image.new('RGB', img_size)

    zoom = (0, 0, im.size[0], im.size[1])
    draw = ImageDraw.Draw(im)

    render(code, zoom, draw)

    del draw
    return im


def render_to_file(filename, img_size, code):

    im = render_to_img(img_size, code)
    im.save(filename, 'PNG')


if __name__ == '__main__':
    main()

import imagehash  # Using https://pypi.python.org/pypi/ImageHash
from PIL import Image, ImageDraw  # Using https://pillow.readthedocs.io

import i2c_domain


Colors = {
    i2c_domain.W: (255, 255, 255),
    i2c_domain.B: (0, 0, 0),
    i2c_domain.G: (128, 128, 128),
    i2c_domain.Re: (255, 0, 0),
    i2c_domain.Gr: (0, 255, 0),
    i2c_domain.Bl: (0, 0, 255)
}

def render(code, zoom, draw):
    if isinstance(code, list):
        func_name = code[0]

        if is_split_operator(func_name):

            new_zooms = split_zoom(func_name, zoom)

            if len(new_zooms) != len(code) - 1:
                raise ValueError('Split operator mus have 2 args.')

            for i in range(0, len(new_zooms)):
                render(code[i + 1], new_zooms[i], draw)

        elif is_color_encoding(func_name):

            if len(code) != 4:
                raise ValueError('Color encoding must have 3 args.')

            render_color((code[1], code[2], code[3]), zoom, draw)

        else:
            raise ValueError('Unsupported function', func_name)

    elif isinstance(code, str):
        render_color(Colors[code], zoom, draw)

    else:
        raise ValueError("Unsupported code format.")


def render_color(color, zoom, draw):
    draw.rectangle(zoom, fill=color)


def is_split_operator(func_name):
    return func_name == i2c_domain.H or func_name == i2c_domain.V or func_name == i2c_domain.Q


def is_color_encoding(func_name):
    return func_name == i2c_domain.C


def split_zoom(func_name, zoom):
    x1, y1, x2, y2 = zoom
    if func_name == i2c_domain.H:
        y_split = round((y1 + y2) / 2)
        return [(x1, y1, x2, y_split), (x1, y_split, x2, y2)]
    elif func_name == i2c_domain.V:
        x_split = round((x1 + x2) / 2)
        return [(x1, y1, x_split, y2), (x_split, y1, x2, y2)]
    elif func_name == i2c_domain.Q:
        x_split = round((x1 + x2) / 2)
        y_split = round((y1 + y2) / 2)
        return [
            (x1, y1, x_split, y_split), (x_split, y1, x2, y_split),
            (x1, y_split, x_split, y2), (x_split, y_split, x2, y2)
        ]
    else:
        raise ValueError("Unsupported function", func_name)


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

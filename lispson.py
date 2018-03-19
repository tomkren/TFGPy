

langs = {
    'python': {
    },
    'javascript': {
        'bool': {
            'True': 'true',
            'False': 'false'
        }
    }
}


def decode_bool(b, lib):
    b_str = str(b)
    if 'bool' in lib['lang']:
        return lib['lang']['bool'][b_str]
    else:
        return b_str




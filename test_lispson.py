import unittest

import lispson

langs_bool = {
    'python': {
    },
    'javascript': {
        'bool': {
            'True': 'true',
            'False': 'false'
        }
    }
}


class TestLispson(unittest.TestCase):

    def check_decoding(self, langs, lang_name, to_decode, correct_answer):
        python_lib = {'lang': langs[lang_name]}
        answer = lispson.decode_bool(to_decode, python_lib)
        self.assertEqual(answer, correct_answer)

    def test_decode_bool_python(self):
        self.check_decoding(langs_bool, 'python', True, 'True')
        self.check_decoding(langs_bool, 'python', not True, 'False')
        self.check_decoding(langs_bool, 'python', False, 'False')
        self.check_decoding(langs_bool, 'python', 'True', 'True')
        self.check_decoding(langs_bool, 'python', 'not True', 'not True')
        self.check_decoding(langs_bool, 'python', 'False', 'False')

    def test_decode_bool_js(self):
        self.check_decoding(langs_bool, 'javascript', True, 'true')
        self.check_decoding(langs_bool, 'javascript', not True, 'false')
        self.check_decoding(langs_bool, 'javascript', False, 'false')
        self.check_decoding(langs_bool, 'javascript', 'True', 'true')


if __name__ == "__main__":
    unittest.main()

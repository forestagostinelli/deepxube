import unittest
from deepxube.logic.logic_utils import parse_literal


class TestStringMethods(unittest.TestCase):

    def test_parse_lit_arity0(self):
        self.assertEqual(parse_literal("lit").to_code(), 'lit')
        self.assertEqual(parse_literal("lit ").to_code(), 'lit')
        self.assertEqual(parse_literal(" lit").to_code(), 'lit')
        self.assertEqual(parse_literal(" lit ").to_code(), 'lit')

    def test_parse_lit_arity0_not(self):
        self.assertEqual(parse_literal("not lit").to_code(), 'not lit')
        self.assertEqual(parse_literal(" not lit ").to_code(), 'not lit')
        self.assertEqual(parse_literal(" not  lit ").to_code(), 'not lit')

    def test_parse_lit_arity1(self):
        self.assertEqual(parse_literal("lit(A)").to_code(), 'lit(A)')
        self.assertEqual(parse_literal(" lit( A ) ").to_code(), 'lit(A)')

    def test_parse_lit_arity1_not(self):
        self.assertEqual(parse_literal("not lit(A)").to_code(), 'not lit(A)')
        self.assertEqual(parse_literal(" not  lit( A ) ").to_code(), 'not lit(A)')

    def test_parse_lit_arity2(self):
        self.assertEqual(parse_literal("lit(A,B)").to_code(), 'lit(A,B)')
        self.assertEqual(parse_literal("lit( A, B )").to_code(), 'lit(A,B)')

    def test_parse_lit_arity2_not(self):
        self.assertEqual(parse_literal("not lit(A,B)").to_code(), 'not lit(A,B)')
        self.assertEqual(parse_literal("not lit( A, B )").to_code(), 'not lit(A,B)')


if __name__ == '__main__':
    unittest.main()

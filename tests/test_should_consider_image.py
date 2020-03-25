import unittest
from modules.utils.helpers import should_consider_image


class TestShouldConsiderImage(unittest.TestCase):
    def test_section_0(self):
        section = 0
        left_back = should_consider_image(
            section, {'LeftBack': 'True', 'LeftFront': 'False', 'LeftLeft': 'False'}
        )
        left_left = should_consider_image(
            section, {'LeftBack': 'False', 'LeftFront': 'False', 'LeftLeft': 'True'}
        )
        left_front = should_consider_image(
            section, {'LeftBack': 'False', 'LeftFront': 'True', 'LeftLeft': 'False'}
        )
        self.assertEqual(left_back, True)
        self.assertEqual(left_left, False)
        self.assertEqual(left_front, False)

    def test_section_1(self):
        section = 1
        left_back = should_consider_image(
            section, {'LeftBack': 'True', 'LeftFront': 'False', 'LeftLeft': 'False'}
        )
        left_left = should_consider_image(
            section, {'LeftBack': 'False', 'LeftFront': 'False', 'LeftLeft': 'True'}
        )
        left_front = should_consider_image(
            section, {'LeftBack': 'False', 'LeftFront': 'True', 'LeftLeft': 'False'}
        )
        self.assertEqual(left_back, False)
        self.assertEqual(left_left, True)
        self.assertEqual(left_front, False)

    def test_section_2(self):
        section = 2
        left_back = should_consider_image(
            section, {'LeftBack': 'True', 'LeftFront': 'False', 'LeftLeft': 'False'}
        )
        left_left = should_consider_image(
            section, {'LeftBack': 'False', 'LeftFront': 'False', 'LeftLeft': 'True'}
        )
        left_front = should_consider_image(
            section, {'LeftBack': 'False', 'LeftFront': 'True', 'LeftLeft': 'False'}
        )
        self.assertEqual(left_back, False)
        self.assertEqual(left_left, False)
        self.assertEqual(left_front, True)


if __name__ == '__main__':
    unittest.main()

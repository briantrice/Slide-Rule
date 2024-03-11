import math
import unittest

from SlideRule import scale_linear, scale_pythagorean, scale_square, scale_cube, SCALE_CONFIGS, scale_sqrt, scale_log


class ScaleLinearTestCase(unittest.TestCase):
    def test_fenceposts(self):
        self.assertEqual(scale_linear(1), 0)
        self.assertAlmostEqual(scale_linear(math.pi), 0.49714987269413385)
        self.assertEqual(scale_linear(math.sqrt(10)), 0.5)
        self.assertEqual(scale_linear(10), 1)


class ScaleLogTestCase(unittest.TestCase):
    def test_fenceposts(self):
        self.assertEqual(scale_log(0), 0)
        self.assertEqual(scale_log(1), 0.1)
        self.assertEqual(scale_log(2), 0.2)
        self.assertEqual(scale_log(3), 0.3)
        self.assertEqual(scale_log(4), 0.4)
        self.assertEqual(scale_log(5), 0.5)
        self.assertEqual(scale_log(6), 0.6)
        self.assertEqual(scale_log(7), 0.7)
        self.assertEqual(scale_log(8), 0.8)
        self.assertEqual(scale_log(9), 0.9)
        self.assertEqual(scale_log(10), 1)


class ScaleLinearPiFoldedTestCase(unittest.TestCase):
    def test_fenceposts(self):
        cf_scale = SCALE_CONFIGS['CF']

        self.assertEqual(cf_scale.frac_pos_of(math.pi), 1)
        self.assertAlmostEqual(cf_scale.frac_pos_of(math.pi / 10), 0)


class ScaleSquareTestCase(unittest.TestCase):
    def test_fenceposts(self):
        self.assertEqual(scale_square(1), 0)
        self.assertEqual(scale_square(10), 0.5)
        self.assertEqual(scale_square(100), 1)

    def test_against_linear(self):
        self.assertEqual(scale_square(47), scale_linear(math.sqrt(47)))
        self.assertEqual(scale_square(16), scale_linear(4))


class ScaleSqrtTestCase(unittest.TestCase):
    def xtest_fenceposts(self):
        self.assertEqual(scale_sqrt(1), scale_linear(1))
        self.assertEqual(scale_sqrt(math.sqrt(10)), scale_linear(10))


class ScaleSqrtTenTestCase(unittest.TestCase):
    def xtest_fenceposts(self):
        pass


class ScaleCubeTestCase(unittest.TestCase):
    def test_fenceposts(self):
        self.assertEqual(scale_cube(1), 0)
        self.assertEqual(scale_cube(10), 1/3)
        self.assertEqual(scale_cube(100), 2/3)
        self.assertEqual(scale_cube(1000), 1)

    def test_against_linear(self):
        self.assertEqual(scale_cube(2**3), scale_linear(2))
        self.assertAlmostEqual(scale_cube(5**3), scale_linear(5))


class ScalePythagoreanTestCase(unittest.TestCase):
    def test_top(self):
        self.assertEqual(scale_pythagorean(0), scale_linear(10))

    def test_scale_at_45deg(self):
        x_rad = math.radians(45)
        x_cos = math.cos(x_rad)
        self.assertAlmostEqual(scale_pythagorean(x_cos), scale_linear(x_cos * 10))

    def test_fenceposts(self):
        self.assertAlmostEqual(scale_pythagorean(0.8), scale_linear(6))
        self.assertAlmostEqual(scale_pythagorean(0.6), scale_linear(8))

    def test_against_linear(self):
        self.assertEqual(scale_pythagorean(0), scale_linear(10))
        self.assertAlmostEqual(scale_pythagorean(0.9), scale_linear(4.3588985))
        self.assertAlmostEqual(scale_pythagorean(0.8), scale_linear(6))
        self.assertAlmostEqual(scale_pythagorean(0.7), scale_linear(7.141428))
        self.assertAlmostEqual(scale_pythagorean(0.6), scale_linear(8))
        self.assertAlmostEqual(scale_pythagorean(0.5), scale_linear(8.660255))
        self.assertAlmostEqual(scale_pythagorean(0.4), scale_linear(9.165152))
        self.assertAlmostEqual(scale_pythagorean(0.3), scale_linear(9.539393))
        self.assertAlmostEqual(scale_pythagorean(0.2), scale_linear(9.797958))

    def test_bottom(self):
        bottom = 0.99498743710662
        self.assertAlmostEqual(bottom, math.sqrt(1 - 0.1**2))
        self.assertAlmostEqual(scale_pythagorean(bottom), 0)


if __name__ == '__main__':
    unittest.main()

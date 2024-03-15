import math
import unittest

from SlideRule import scale_base, scale_pythagorean, scale_square, scale_cube, scale_sqrt, scale_log, \
    scale_log_log2, Scales, scale_log_log1, scale_log_log3, scale_log_log03, scale_log_log02, scale_log_log01, \
    symbol_with_expon


class ScaleBaseTestCase(unittest.TestCase):
    def test_fenceposts(self):
        self.assertEqual(scale_base(1), 0)
        self.assertAlmostEqual(scale_base(math.pi), 0.49714987269413385)
        self.assertEqual(scale_base(math.sqrt(10)), 0.5)
        self.assertEqual(scale_base(10), 1)


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
        self.assertEqual(scale_log(10), 1.0)


class ScaleBasePiFoldedTestCase(unittest.TestCase):
    def test_fenceposts(self):
        cf_scale = Scales.CF

        self.assertEqual(cf_scale.frac_pos_of(math.pi), 1)
        self.assertAlmostEqual(cf_scale.frac_pos_of(math.pi / 10), 0)


class ScaleSquareTestCase(unittest.TestCase):
    def test_fenceposts(self):
        self.assertEqual(scale_square(1), 0)
        self.assertEqual(scale_square(10), 0.5)
        self.assertEqual(scale_square(100), 1)

    def test_against_base(self):
        self.assertEqual(scale_square(47), scale_base(math.sqrt(47)))
        self.assertEqual(scale_square(16), scale_base(4))


class ScaleSqrtTestCase(unittest.TestCase):
    def xtest_fenceposts(self):
        self.assertEqual(scale_sqrt(1), scale_base(1))
        self.assertEqual(scale_sqrt(math.sqrt(10)), scale_base(10))


class ScaleLogLogTestCase(unittest.TestCase):
    def test_fenceposts(self):
        self.assertAlmostEqual(scale_log_log1(math.e), 2)
        self.assertAlmostEqual(scale_log_log2(math.e), 1)
        self.assertAlmostEqual(scale_log_log3(math.e), 0)

    def test_ends(self):
        self.assertAlmostEqual(scale_log_log3(math.e), 0)
        self.assertAlmostEqual(scale_log_log3(22026), 1, 5)
        self.assertAlmostEqual(scale_log_log2(1.105171), 0, 5)
        self.assertAlmostEqual(scale_log_log2(math.e), 1)
        self.assertAlmostEqual(scale_log_log1(1.0100501), 0, 5)
        self.assertAlmostEqual(scale_log_log1(1.105171), 1, 5)
        self.assertAlmostEqual(scale_log_log03(1/math.e), 0)
        self.assertAlmostEqual(scale_log_log03(0.0000454), 1, 5)
        self.assertAlmostEqual(scale_log_log02(0.904837), 0, 5)
        self.assertAlmostEqual(scale_log_log02(1/math.e), 1)
        self.assertAlmostEqual(scale_log_log01(0.9900498), 0, 5)
        self.assertAlmostEqual(scale_log_log01(0.904837), 1, 5)


class ScaleSqrtTenTestCase(unittest.TestCase):
    def xtest_fenceposts(self):
        pass


class ScaleCubeTestCase(unittest.TestCase):
    def test_fenceposts(self):
        self.assertEqual(scale_cube(1), 0)
        self.assertEqual(scale_cube(10), 1/3)
        self.assertEqual(scale_cube(100), 2/3)
        self.assertEqual(scale_cube(1000), 1)

    def test_against_base(self):
        self.assertEqual(scale_cube(2**3), scale_base(2))
        self.assertAlmostEqual(scale_cube(5**3), scale_base(5))


class ScalePythagoreanTestCase(unittest.TestCase):
    def test_top(self):
        self.assertEqual(scale_pythagorean(0), scale_base(10))

    def test_scale_at_45deg(self):
        x_rad = math.radians(45)
        x_cos = math.cos(x_rad)
        self.assertAlmostEqual(scale_pythagorean(x_cos), scale_base(x_cos * 10))

    def test_fenceposts(self):
        self.assertAlmostEqual(scale_pythagorean(0.8), scale_base(6))
        self.assertAlmostEqual(scale_pythagorean(0.6), scale_base(8))

    def test_against_base(self):
        self.assertEqual(scale_pythagorean(0), scale_base(10))
        self.assertAlmostEqual(scale_pythagorean(0.9), scale_base(4.3588985))
        self.assertAlmostEqual(scale_pythagorean(0.8), scale_base(6))
        self.assertAlmostEqual(scale_pythagorean(0.7), scale_base(7.141428))
        self.assertAlmostEqual(scale_pythagorean(0.6), scale_base(8))
        self.assertAlmostEqual(scale_pythagorean(0.5), scale_base(8.660255))
        self.assertAlmostEqual(scale_pythagorean(0.4), scale_base(9.165152))
        self.assertAlmostEqual(scale_pythagorean(0.3), scale_base(9.539393))
        self.assertAlmostEqual(scale_pythagorean(0.2), scale_base(9.797958))

    def test_bottom(self):
        bottom = 0.99498743710662
        self.assertAlmostEqual(bottom, math.sqrt(1 - 0.1**2))
        self.assertAlmostEqual(scale_pythagorean(bottom), 0)


class UtilsTestCase(unittest.TestCase):
    def test_symbol_with_expon_caret(self):
        self.assertEqual(('x', 'y'), symbol_with_expon('x^y'))
        self.assertEqual(('10', '4'), symbol_with_expon('10^4'))
        self.assertEqual(('10', '-3'), symbol_with_expon('10^-3'))
        self.assertEqual(('10', '-3.4'), symbol_with_expon('10^-3.4'))
        self.assertEqual(('10', '0'), symbol_with_expon('10^0'))

    def test_symbol_with_expon_unicode(self):
        self.assertEqual(('10', '4'), symbol_with_expon('10⁴'))
        self.assertEqual(('10', '-4'), symbol_with_expon('10⁻⁴'))
        self.assertEqual(('10', '04'), symbol_with_expon('10⁰⁴'))


if __name__ == '__main__':
    unittest.main()

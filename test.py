import math
import unittest

from SlideRule import scale_sqrt, \
    scale_log_log2, Scales, scale_log_log1, scale_log_log3, scale_log_log03, scale_log_log02, scale_log_log01, \
    symbol_with_expon, scale_sqrt_ten, scale_hyperbolic, Scalers, symbol_parts

scale_base = Scalers.Base


class ScaleBaseTestCase(unittest.TestCase):
    def test_fenceposts(self):
        self.assertEqual(scale_base(1), 0)
        self.assertAlmostEqual(scale_base(math.pi), 0.49714987269413385)
        self.assertEqual(scale_base(math.sqrt(10)), 0.5)
        self.assertEqual(scale_base(10), 1)

    def test_value_at(self):
        self.assertEqual(scale_base.value_at(0), 1)
        self.assertEqual(scale_base.value_at(1), 10)


class ScaleLogTestCase(unittest.TestCase):
    def test_fenceposts(self):
        scale_log = Scalers.Log10
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

    def test_value_at(self):
        s = Scalers.Log10
        self.assertEqual(s.value_at(0), 0)
        self.assertEqual(s.value_at(1), 10)
        self.assertEqual(s.value_at(0.1), 1)


class ScaleBasePiFoldedTestCase(unittest.TestCase):
    def test_fenceposts(self):
        cf_scale = Scales.CF

        self.assertEqual(cf_scale.frac_pos_of(math.pi), 1)
        self.assertAlmostEqual(cf_scale.frac_pos_of(math.pi / 10), 0)


class ScaleSquareTestCase(unittest.TestCase):
    def test_fenceposts(self):
        scale_square = Scalers.Square
        self.assertEqual(scale_square(1), 0)
        self.assertEqual(scale_square(10), 0.5)
        self.assertEqual(scale_square(100), 1)

    def test_against_base(self):
        scale_square = Scalers.Square
        self.assertEqual(scale_square(47), scale_base(math.sqrt(47)))
        self.assertEqual(scale_square(16), scale_base(4))

    def test_value_at(self):
        s = Scalers.Square
        self.assertEqual(s.value_at(0), 1)
        self.assertEqual(s.value_at(0.25), math.sqrt(10))
        self.assertEqual(s.value_at(0.5), 10)
        self.assertEqual(s.value_at(0.75), math.sqrt(1000))
        self.assertEqual(s.value_at(1), 100)


class ScaleSqrtTestCase(unittest.TestCase):
    def test_fenceposts(self):
        self.assertEqual(scale_sqrt(1), -2)
        self.assertEqual(scale_sqrt(math.sqrt(10)), -1)
        self.assertEqual(scale_sqrt(10), 0)
        self.assertEqual(scale_sqrt(100), 2)
        self.assertEqual(scale_sqrt(1000), 4)


class ScaleHyperbolicTestCase(unittest.TestCase):
    def test_fenceposts(self):
        self.assertAlmostEqual(scale_hyperbolic(1.4142135), scale_base(1))
        self.assertAlmostEqual(scale_hyperbolic(10.049875), scale_base(10))


class ScaleThetaTestCase(unittest.TestCase):
    def test_fenceposts(self):
        self.assertEqual(Scales.Theta.frac_pos_of(0), 0)
        self.assertEqual(Scales.Theta.frac_pos_of(90), 1)


class ScaleSqrtTenTestCase(unittest.TestCase):
    def test_fenceposts(self):
        self.assertEqual(scale_sqrt_ten(1), scale_base(1))
        self.assertEqual(scale_sqrt_ten(10), scale_base(10))


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


class ScaleCubeTestCase(unittest.TestCase):
    def test_fenceposts(self):
        scale_cube = Scalers.Cube
        self.assertEqual(scale_cube(1), 0)
        self.assertEqual(scale_cube(10), 1/3)
        self.assertEqual(scale_cube(100), 2/3)
        self.assertEqual(scale_cube(1000), 1)

    def test_against_base(self):
        scale_cube = Scalers.Cube
        self.assertEqual(scale_cube(2**3), scale_base(2))
        self.assertAlmostEqual(scale_cube(5**3), scale_base(5))

    def test_value_at(self):
        s = Scalers.Cube
        self.assertEqual(s.value_at(0), 1)
        self.assertAlmostEqual(s.value_at(1/3), 10)
        self.assertAlmostEqual(s.value_at(2/3), 100)
        self.assertEqual(s.value_at(1), 1000)


class ScalePythagoreanTestCase(unittest.TestCase):
    def test_top(self):
        scale_pythagorean = Scalers.Pythagorean
        self.assertEqual(scale_pythagorean(0), scale_base(10))

    def test_scale_at_45deg(self):
        scale_pythagorean = Scalers.Pythagorean
        x_rad = math.radians(45)
        x_cos = math.cos(x_rad)
        self.assertAlmostEqual(scale_pythagorean(x_cos), scale_base(x_cos * 10))

    def test_fenceposts(self):
        scale_pythagorean = Scalers.Pythagorean
        self.assertAlmostEqual(scale_pythagorean(0.8), scale_base(6))
        self.assertAlmostEqual(scale_pythagorean(0.6), scale_base(8))

    def test_against_base(self):
        scale_pythagorean = Scalers.Pythagorean
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
        scale_pythagorean = Scalers.Pythagorean
        bottom = 0.99498743710662
        self.assertAlmostEqual(bottom, math.sqrt(1 - 0.1**2))
        self.assertAlmostEqual(scale_pythagorean(bottom), 0)

    def test_value_at(self):
        scale_pythagorean = Scalers.Pythagorean
        self.assertAlmostEqual(scale_pythagorean.value_at(0), 0.99498743710662)
        self.assertAlmostEqual(scale_pythagorean.value_at(1), 0)


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

    def test_symbol_with_expon_subscript(self):
        self.assertEqual(('x', 'y', 'z'), symbol_parts('x^y_z'))


if __name__ == '__main__':
    unittest.main()

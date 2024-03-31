import math
import unittest

from SlideRule import (Scales, Scalers, Layout,
                       symbol_parts, symbol_with_expon,
                       scale_hyperbolic, Models)

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

    def test_frac_pos_of(self):
        self.assertEqual(scale_base.position_of(1), 0)
        self.assertEqual(scale_base.position_of(10), 1)


class ScaleInverseTestCase(unittest.TestCase):
    def test_fenceposts(self):
        ci = Scales.CI
        self.assertEqual(ci.value_at_start(), 10)
        self.assertEqual(ci.value_at_end(), 1)

    def test_frac_pos_of(self):
        ci = Scales.CI
        self.assertEqual(ci.frac_pos_of(ci.value_at_start()), 0)
        self.assertEqual(ci.frac_pos_of(ci.value_at_end()), 1)


class ScaleSquareRootTestCase(unittest.TestCase):
    def test_fenceposts(self):
        w1 = Scales.W1
        self.assertEqual(w1.value_at_start(), 1)
        self.assertEqual(w1.value_at_end(), math.sqrt(10))

    def test_frac_pos_of(self):
        w1 = Scales.W1
        self.assertEqual(w1.frac_pos_of(w1.value_at_start()), 0)
        self.assertEqual(w1.frac_pos_of(w1.value_at_end()), 1)
        self.assertAlmostEqual(w1.frac_pos_of(2), Scales.C.frac_pos_of(4))
        self.assertAlmostEqual(w1.frac_pos_of(3), Scales.C.frac_pos_of(9))


class ScaleSqrtTenTestCase(unittest.TestCase):
    def test_fenceposts(self):
        w2 = Scales.W2
        self.assertEqual(w2.value_at_start(), math.sqrt(10))
        self.assertEqual(w2.value_at_end(), 10)


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
        scale_sqrt = Scalers.SquareRoot
        self.assertEqual(scale_sqrt(1), 0)
        self.assertEqual(scale_sqrt(math.sqrt(10)), 1)
        self.assertEqual(scale_sqrt(10), 2)
        self.assertEqual(scale_sqrt(100), 4)
        self.assertEqual(scale_sqrt(1000), 6)


class ScaleHyperbolicTestCase(unittest.TestCase):
    def test_fenceposts(self):
        self.assertAlmostEqual(scale_hyperbolic(1.4142135), scale_base(1))
        self.assertAlmostEqual(scale_hyperbolic(10.049875), scale_base(10))


class ScaleThetaTestCase(unittest.TestCase):
    def test_fenceposts(self):
        self.assertEqual(Scales.Theta.frac_pos_of(0), 0)
        self.assertEqual(Scales.Theta.frac_pos_of(90), 1)


class ScaleLogLogTestCase(unittest.TestCase):
    def test_fenceposts(self):
        self.assertAlmostEqual(Scalers.LogLog.position_of(math.e), 0)

    def test_ends(self):
        self.assertAlmostEqual(Scalers.LogLog.position_of(math.e), 0)
        self.assertAlmostEqual(Scalers.LogLog.position_of(22026), 1, 5)
        self.assertAlmostEqual(Scalers.LogLog.position_of(1.105171), -1, 5)
        self.assertAlmostEqual(Scalers.LogLog.position_of(1.0100501), -2, 5)
        self.assertAlmostEqual(Scalers.LogLog.position_of(1.105171), -1, 5)
        # self.assertAlmostEqual(Scalers.LogLog.position_of(1/math.e), 0)
        # self.assertAlmostEqual(Scalers.LogLog.position_of(0.0000454), 1, 5)
        # self.assertAlmostEqual(Scalers.LogLog.position_of(0.904837), 0, 5)
        # self.assertAlmostEqual(Scalers.LogLog.position_of(0.9900498), 0, 5)
        # self.assertAlmostEqual(Scalers.LogLog.position_of(0.904837), 1, 5)


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


class SlideRuleLayoutTestCase(unittest.TestCase):
    def test_demo(self):
        actual = Layout('|  K,  A  [ B, T, ST, S ] D,  DI    |',
                                 '|  L,  DF [ CF,CIF,CI,C ] D, R1, R2 |')
        self.assertEqual(actual.front_layout, [['K', 'A'], ['B', 'T', 'ST', 'S'], ['D', 'DI']])
        self.assertEqual(actual.rear_layout, [['L', 'DF'], ['CF', 'CIF', 'CI', 'C'], ['D', 'R1', 'R2']])

    def test_single_side_slide(self):
        actual = Layout('A/B C/D')
        self.assertEqual(actual.front_layout, [['A'], ['B', 'C'], ['D']])
        self.assertEqual(actual.rear_layout, [None, None, None])

    def test_no_slide(self):
        actual = Layout('A B C D')
        self.assertEqual(actual.front_layout, [['A', 'B', 'C', 'D'], None, None])
        self.assertEqual(actual.rear_layout, [None, None, None])


class ModelTestCase(unittest.TestCase):
    def test_pickett_n515t_lr(self):
        lr = Scales.L_r
        start_value = 0.025330295910584444
        self.assertEqual(lr.value_at_start(), start_value)
        self.assertEqual(lr.value_at_frac_pos(0.5), start_value / 10)
        self.assertEqual(lr.value_at_end(), start_value / 100)

    def test_pickett_n515t_fx(self):
        fx = Scales.f_x
        self.assertAlmostEqual(fx.value_at_start(), 1 / math.tau)
        self.assertAlmostEqual(fx.value_at_end(), 10 / math.tau)

    def test_faber_castell_283(self):
        fc283 = Models.FaberCastell283
        self.assertEqual(fc283.layout.front_layout, [
            ['K', 'T1', 'T2', 'DF'],
            ['CF', 'CIF', 'CI', 'C'],
            ['D', 'S', 'ST', 'P']])
        self.assertEqual(fc283.layout.rear_layout, [
            ['LL03', 'LL02', 'LL01', 'W2'],
            ['W2Prime', 'L', 'C', 'W1Prime'],
            ['W1', 'LL1', 'LL2', 'LL3']])


if __name__ == '__main__':
    unittest.main()

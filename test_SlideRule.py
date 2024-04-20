import math
import unittest
from dataclasses import replace

from SlideRule import (Scales, ScaleFNs, Layout, Models, RulePart,
                       symbol_parts, symbol_with_expon,
                       last_digit_of, first_digit_of, keys_of, render_diagnostic_mode, Side)

scale_base = ScaleFNs.Base


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
        scale_log = ScaleFNs.Log10
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
        s = ScaleFNs.Log10
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
        scale_square = ScaleFNs.Square
        self.assertEqual(scale_square(1), 0)
        self.assertEqual(scale_square(10), 0.5)
        self.assertEqual(scale_square(100), 1)

    def test_against_base(self):
        scale_square = ScaleFNs.Square
        self.assertEqual(scale_square(47), scale_base(math.sqrt(47)))
        self.assertEqual(scale_square(16), scale_base(4))

    def test_value_at(self):
        s = ScaleFNs.Square
        self.assertEqual(s.value_at(0), 1)
        self.assertEqual(s.value_at(0.25), math.sqrt(10))
        self.assertEqual(s.value_at(0.5), 10)
        self.assertEqual(s.value_at(0.75), math.sqrt(1000))
        self.assertEqual(s.value_at(1), 100)


class ScaleSqrtTestCase(unittest.TestCase):
    def test_fenceposts(self):
        scale_sqrt = ScaleFNs.SquareRoot
        self.assertEqual(scale_sqrt(1), 0)
        self.assertEqual(scale_sqrt(math.sqrt(10)), 1)
        self.assertEqual(scale_sqrt(10), 2)
        self.assertEqual(scale_sqrt(100), 4)
        self.assertEqual(scale_sqrt(1000), 6)


class ScaleHyperbolicTestCase(unittest.TestCase):
    def test_fenceposts(self):
        self.assertAlmostEqual(ScaleFNs.Hyperbolic(1.4142135), scale_base(1))
        self.assertAlmostEqual(ScaleFNs.Hyperbolic(10.049875), scale_base(10))


class ScaleThetaTestCase(unittest.TestCase):
    def test_fenceposts(self):
        self.assertEqual(Scales.Theta.frac_pos_of(0), 0)
        self.assertEqual(Scales.Theta.frac_pos_of(90), 1)


class ScaleLogLogTestCase(unittest.TestCase):
    def test_fenceposts(self):
        self.assertAlmostEqual(ScaleFNs.LogLog.position_of(math.e), 0)

    def test_ends(self):
        self.assertAlmostEqual(ScaleFNs.LogLog.position_of(math.e), 0)
        self.assertAlmostEqual(ScaleFNs.LogLog.position_of(22026), 1, 5)
        self.assertAlmostEqual(ScaleFNs.LogLog.position_of(1.105171), -1, 5)
        self.assertAlmostEqual(ScaleFNs.LogLog.position_of(1.0100501), -2, 5)
        self.assertAlmostEqual(ScaleFNs.LogLog.position_of(1.105171), -1, 5)
        self.assertAlmostEqual(ScaleFNs.LogLogNeg.position_of(1 / math.e), 0)
        self.assertAlmostEqual(ScaleFNs.LogLogNeg.position_of(0.0000454), 1, 5)
        self.assertAlmostEqual(ScaleFNs.LogLogNeg.position_of(0.904837), -1, 5)
        self.assertAlmostEqual(ScaleFNs.LogLogNeg.position_of(0.9900498), -2, 5)
        self.assertAlmostEqual(ScaleFNs.LogLogNeg.position_of(0.904837), -1, 5)


class ScaleCubeTestCase(unittest.TestCase):
    def test_fenceposts(self):
        scale_cube = ScaleFNs.Cube
        self.assertEqual(scale_cube(1), 0)
        self.assertEqual(scale_cube(10), 1/3)
        self.assertEqual(scale_cube(100), 2/3)
        self.assertEqual(scale_cube(1000), 1)

    def test_against_base(self):
        scale_cube = ScaleFNs.Cube
        self.assertEqual(scale_cube(2**3), scale_base(2))
        self.assertAlmostEqual(scale_cube(5**3), scale_base(5))

    def test_value_at(self):
        s = ScaleFNs.Cube
        self.assertEqual(s.value_at(0), 1)
        self.assertAlmostEqual(s.value_at(1/3), 10)
        self.assertAlmostEqual(s.value_at(2/3), 100)
        self.assertEqual(s.value_at(1), 1000)


class ScalePythagoreanTestCase(unittest.TestCase):
    def test_top(self):
        scale_pythagorean = ScaleFNs.Pythagorean
        self.assertEqual(scale_pythagorean(0), scale_base(10))

    def test_scale_at_45deg(self):
        scale_pythagorean = ScaleFNs.Pythagorean
        x_rad = math.radians(45)
        x_cos = math.cos(x_rad)
        self.assertAlmostEqual(scale_pythagorean(x_cos), scale_base(x_cos * 10))

    def test_fenceposts(self):
        scale_pythagorean = ScaleFNs.Pythagorean
        self.assertAlmostEqual(scale_pythagorean(0.8), scale_base(6))
        self.assertAlmostEqual(scale_pythagorean(0.6), scale_base(8))

    def test_against_base(self):
        scale_pythagorean = ScaleFNs.Pythagorean
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
        scale_pythagorean = ScaleFNs.Pythagorean
        bottom = 0.99498743710662
        self.assertAlmostEqual(bottom, math.sqrt(1 - 0.1**2))
        self.assertAlmostEqual(scale_pythagorean(bottom), 0)

    def test_value_at(self):
        scale_pythagorean = ScaleFNs.Pythagorean
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

    def test_symbol_with_expon_prime(self):
        self.assertEqual(('W', "'"), symbol_with_expon("W'"))
        self.assertEqual(('x', "'"), symbol_with_expon("x'"))

    def test_symbol_with_expon_subscript(self):
        self.assertEqual(('x', 'y', 'z'), symbol_parts('x^y_z'))
        self.assertEqual(('W', "'", '2'), symbol_parts("W'₂"))

    def test_last_digit_of(self):
        self.assertEqual(5, last_digit_of(5))
        self.assertEqual(5, last_digit_of(15))
        self.assertEqual(5, last_digit_of(65))
        self.assertEqual(5, last_digit_of(105))
        self.assertEqual(5, last_digit_of(0.5))
        self.assertEqual(5, last_digit_of(0.15))
        self.assertEqual(5, last_digit_of(0.65))
        self.assertEqual(5, last_digit_of(10.5))
        self.assertEqual(1, last_digit_of(11.0))

    def test_first_digit_of(self):
        self.assertEqual(5, first_digit_of(5))
        self.assertEqual(1, first_digit_of(15))
        self.assertEqual(6, first_digit_of(65))
        self.assertEqual(1, first_digit_of(105))


class LayoutTestCase(unittest.TestCase):
    def test_demo(self):
        actual = Layout('|  K,  A  [ B, T, ST, S ] D,  DI    |',
                        '|  L,  DF [ CF,CIF,CI,C ] D, R1, R2 |')
        self.assertListEqual(actual.sc_keys_at(Side.FRONT, RulePart.STATOR_TOP), ['K', 'A'])
        self.assertListEqual(actual.sc_keys_at(Side.FRONT, RulePart.SLIDE), ['B', 'T', 'ST', 'S'])
        self.assertListEqual(actual.sc_keys_at(Side.FRONT, RulePart.STATOR_BOTTOM), ['D', 'DI'])
        self.assertListEqual(actual.sc_keys_at(Side.REAR, RulePart.STATOR_TOP), ['L', 'DF'])
        self.assertListEqual(actual.sc_keys_at(Side.REAR, RulePart.SLIDE), ['CF', 'CIF', 'CI', 'C'])
        self.assertListEqual(actual.sc_keys_at(Side.REAR, RulePart.STATOR_BOTTOM), ['D', 'R1', 'R2'])

    def test_single_side_slide(self):
        actual = Layout('A/B C/D', '')
        self.assertListEqual(actual.sc_keys_at(Side.FRONT, RulePart.STATOR_TOP), ['A'])
        self.assertListEqual(actual.sc_keys_at(Side.FRONT, RulePart.SLIDE), ['B', 'C'])
        self.assertListEqual(actual.sc_keys_at(Side.FRONT, RulePart.STATOR_BOTTOM), ['D'])
        self.assertIsNone(actual.sc_keys_at(Side.REAR, RulePart.STATOR_TOP))
        self.assertIsNone(actual.sc_keys_at(Side.REAR, RulePart.SLIDE))
        self.assertIsNone(actual.sc_keys_at(Side.REAR, RulePart.STATOR_BOTTOM))

    def test_no_slide(self):
        actual = Layout('A B C D', '')
        self.assertListEqual(actual.sc_keys_at(Side.FRONT, RulePart.STATOR_TOP), ['A', 'B', 'C', 'D'])
        self.assertIsNone(actual.sc_keys_at(Side.FRONT, RulePart.SLIDE))
        self.assertIsNone(actual.sc_keys_at(Side.FRONT, RulePart.STATOR_BOTTOM))
        self.assertIsNone(actual.sc_keys_at(Side.REAR, RulePart.STATOR_TOP))
        self.assertIsNone(actual.sc_keys_at(Side.REAR, RulePart.SLIDE))
        self.assertIsNone(actual.sc_keys_at(Side.REAR, RulePart.STATOR_BOTTOM))

    def test_slide_only_side(self):
        example = '[S ST T]'
        self.assertEqual(['', 'S ST T', ''], Layout.parts_of_side_layout(example))

        actual1 = Layout.parse_side_layout(example)
        self.assertIsNone(actual1[RulePart.STATOR_TOP])
        self.assertListEqual(actual1[RulePart.SLIDE], ['S', 'ST', 'T'])
        self.assertIsNone(actual1[RulePart.STATOR_BOTTOM])

        actual2 = Layout('A / B C / D', example)
        self.assertListEqual(actual2.sc_keys_at(Side.FRONT, RulePart.STATOR_TOP), ['A'])
        self.assertListEqual(actual2.sc_keys_at(Side.FRONT, RulePart.SLIDE), ['B', 'C'])
        self.assertListEqual(actual2.sc_keys_at(Side.FRONT, RulePart.STATOR_BOTTOM), ['D'])
        self.assertIsNone(actual2.sc_keys_at(Side.REAR, RulePart.STATOR_TOP))
        self.assertListEqual(actual2.sc_keys_at(Side.REAR, RulePart.SLIDE), ['S', 'ST', 'T'])
        self.assertIsNone(actual2.sc_keys_at(Side.REAR, RulePart.STATOR_BOTTOM))


class ModelTestCase(unittest.TestCase):
    def test_pickett_n515t_lr(self):
        lr = Scales.L_r
        start_value = 0.25330295910584444
        self.assertAlmostEqual(lr.value_at_start(), start_value)
        self.assertAlmostEqual(lr.value_at_frac_pos(0.5), start_value / 10)
        self.assertAlmostEqual(lr.value_at_end(), start_value / 100)

    def test_pickett_n515t_fx(self):
        fx = Scales.f_x
        self.assertAlmostEqual(fx.value_at_start(), 1 / math.tau)
        self.assertAlmostEqual(fx.value_at_end(), 10 / math.tau)

    def test_faber_castell_283(self):
        fc283 = Models.FaberCastell283
        self.assertDictEqual(fc283.layout.sc_keys[Side.FRONT], {
            RulePart.STATOR_TOP: ['K', 'T1', 'T2', 'DF'],
            RulePart.SLIDE: ['CF', 'CIF', 'CI', 'C'],
            RulePart.STATOR_BOTTOM: ['D', 'S', 'ST', 'P']})
        self.assertDictEqual(fc283.layout.sc_keys[Side.REAR], {
            RulePart.STATOR_TOP: ['LL03', 'LL02', 'LL01', 'W2'],
            RulePart.SLIDE: ['W2Prime', 'L', 'C', 'W1Prime'],
            RulePart.STATOR_BOTTOM: ['W1', 'LL1', 'LL2', 'LL3']})


if __name__ == '__main__':
    unittest.main()

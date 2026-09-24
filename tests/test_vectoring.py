"""Vectoring-mode CORDIC tests: atan2 and magnitude.

Coverage:

* hand-computed analytic values (axes, diagonals, 3-4-5 triples),
* the independent Decimal oracle ``atan2_mag_ref`` (never calls the core),
* cross-checks against stdlib ``math.atan2``/``math.hypot`` (a second,
  independent source of expected values),
* the zero vector and negative-x-axis conventions,
* INT32_MIN coordinates and the magnitude saturation regime,
* small-box exhaustive enumeration and a deterministic coarse full-range grid,
* input validation and failure semantics (bool, NaN/Inf, range, purity).

The CORDIC core under test is never used to generate its own expectations.
"""

import math
import unittest
from decimal import Decimal

from cordic_q428 import (
    FRAC_BITS,
    INT32_MAX,
    INT32_MIN,
    ITERATIONS,
    PI_Q,
    HALF_PI_Q,
    SCALE,
    Atan2Result,
    CordicError,
    FixedPointOverflowError,
    InvalidArgumentTypeError,
    NonFiniteValueError,
    atan2,
    atan2_mag,
    atan2_mag_radians,
    atan2_mag_ref,
    magnitude,
    mul_q,
    rshift_even,
    saturate_int32,
)

# Safe, non-loose bounds.  Analytically the 32 floor-shifts (each <= 1 ulp)
# amplified by the ~1.647 gain, plus <= 1 ulp gain-table quantization, bound
# the datapath well under 48 ulp; observed maxima are ~4 ulp (angle) and
# ~14 ulp (magnitude at the largest radii).
MAX_ANGLE_ERR_ULP = 48
MAX_MAG_ERR_ULP = 48


def _ae_ulp(r: Atan2Result) -> float:
    return float(r.angle_error) * SCALE


def _me_ulp(r: Atan2Result) -> float:
    return float(r.mag_error) * SCALE


class TestAnalyticValues(unittest.TestCase):
    """Hand-computed values that need neither the core nor the Decimal oracle."""

    def test_zero_vector_convention(self):
        r = atan2_mag(0, 0)
        self.assertEqual(r.angle_q, 0)
        self.assertEqual(r.mag_q, 0)
        self.assertFalse(r.saturated)
        self.assertEqual(r.angle_offset_q, 0)
        # The zero vector performs no rotations: 32 explicit idle states.
        self.assertEqual(len(r.rounds), ITERATIONS)
        self.assertTrue(all(st.direction == 0 for st in r.rounds))
        self.assertTrue(all(st.x == 0 and st.y == 0 and st.z == 0 for st in r.rounds))
        self.assertEqual(r.angle_error, 0)
        self.assertEqual(r.mag_error, 0)

    def test_positive_x_axis(self):
        for k in (1, SCALE, INT32_MAX):
            with self.subTest(k=k):
                r = atan2_mag(0, k)
                self.assertEqual(r.angle_q, 0)
                self.assertEqual(r.angle_offset_q, 0)
                self.assertLessEqual(abs(r.mag_q - k), MAX_MAG_ERR_ULP)

    def test_positive_y_axis(self):
        r = atan2_mag(SCALE, 0)
        self.assertLessEqual(abs(r.angle_q - HALF_PI_Q), MAX_ANGLE_ERR_ULP)
        self.assertLessEqual(abs(r.mag_q - SCALE), MAX_MAG_ERR_ULP)

    def test_negative_y_axis(self):
        r = atan2_mag(-SCALE, 0)
        self.assertLessEqual(abs(r.angle_q + HALF_PI_Q), MAX_ANGLE_ERR_ULP)
        self.assertLessEqual(abs(r.mag_q - SCALE), MAX_MAG_ERR_ULP)

    def test_negative_x_axis_is_plus_pi(self):
        # Integer inputs carry no signed zero; the documented convention for
        # atan2(0, x<0) is +pi (math.atan2(+0.0, negative) agrees).
        r = atan2_mag(0, -SCALE)
        self.assertEqual(r.angle_offset_q, PI_Q)
        self.assertLessEqual(abs(r.angle_q - PI_Q), MAX_ANGLE_ERR_ULP)
        self.assertGreater(r.angle_q, 0)
        self.assertLessEqual(abs(r.mag_q - SCALE), MAX_MAG_ERR_ULP)
        self.assertAlmostEqual(r.angle, math.pi, delta=48 / SCALE)

    def test_four_diagonals(self):
        s = SCALE
        expected = {
            (s, s): math.pi / 4,
            (-s, s): -math.pi / 4,
            (s, -s): 3 * math.pi / 4,
            (-s, -s): -3 * math.pi / 4,
        }
        for (y, x), want in expected.items():
            with self.subTest(y=y, x=x):
                r = atan2_mag(y, x)
                self.assertLessEqual(abs(_ae_ulp(r)), MAX_ANGLE_ERR_ULP)
                self.assertAlmostEqual(r.angle, want, delta=MAX_ANGLE_ERR_ULP / SCALE)
                # |mag - sqrt(2)|; compare through squared identity instead
                # of a hard-coded decimal constant.
                mag = r.mag_q / SCALE
                self.assertAlmostEqual(mag * mag, 2.0, delta=2 * MAX_MAG_ERR_ULP / SCALE)

    def test_three_four_five_triples(self):
        # Integer-coordinate Pythagorean triples: (3,4)->mag 5 exactly in the
        # reals.  Several scales and all sign patterns are exercised.
        for n in (1, SCALE // 100, SCALE // 10):
            for sy in (1, -1):
                for sx in (1, -1):
                    y, x, want_mag = sy * 3 * n, sx * 4 * n, 5 * n
                    if not (INT32_MIN <= y <= INT32_MAX and INT32_MIN <= x <= INT32_MAX):
                        continue
                    with self.subTest(y=y, x=x):
                        r = atan2_mag(y, x)
                        self.assertFalse(r.saturated)
                        self.assertLessEqual(abs(r.mag_q - want_mag), MAX_MAG_ERR_ULP)

    def test_angle_always_in_minus_pi_pi(self):
        for y in (-INT32_MAX, -PI_Q, -1, 0, 1, PI_Q, INT32_MAX):
            for x in (-INT32_MAX, -1, 0, 1, INT32_MAX):
                r = atan2_mag(y, x)
                self.assertTrue(-PI_Q - MAX_ANGLE_ERR_ULP <= r.angle_q <= PI_Q + MAX_ANGLE_ERR_ULP)


class TestAgainstOracles(unittest.TestCase):
    def test_decimal_oracle_matches_stdlib_math(self):
        # Guard the oracle itself with an independent source (stdlib math),
        # never with the core under test.  Compare against math evaluated on
        # the *quantized* coordinates, since the oracle faithfully serves the
        # Q4.28 inputs rather than the pre-quantization floats.
        for yf, xf in ((1.0, 1.0), (-3.0, 2.0), (0.001, -0.002), (-7.0, -7.0), (0.0, -1.0)):
            yq, xq = round(yf * SCALE), round(xf * SCALE)
            a, m = atan2_mag_ref(yq, xq)
            self.assertAlmostEqual(float(a), math.atan2(yq / SCALE, xq / SCALE), delta=1e-9)
            self.assertAlmostEqual(float(m), math.hypot(yq / SCALE, xq / SCALE), delta=1e-9)

    def test_oracle_zero_vector(self):
        a, m = atan2_mag_ref(0, 0)
        self.assertEqual((a, m), (Decimal(0), Decimal(0)))

    def test_core_against_decimal_oracle_small_box(self):
        # Exhaustive enumeration of every vector in [-5, 5]^2: tiny inputs
        # are the regime where normalization matters most.
        for y in range(-5, 6):
            for x in range(-5, 6):
                with self.subTest(y=y, x=x):
                    r = atan2_mag(y, x)
                    self.assertLessEqual(abs(_ae_ulp(r)), MAX_ANGLE_ERR_ULP)
                    if not r.saturated:
                        self.assertLessEqual(abs(_me_ulp(r)), MAX_MAG_ERR_ULP)

    def test_core_against_math_on_ring(self):
        # Deterministic LCG ring of vectors at 8 radii x 256 angles.
        radii = (1, 10, 100, 1000, SCALE // 8, SCALE, 3 * SCALE, 7 * SCALE)
        state = 0x1234_9678
        for idx in range(256 * 8):
            state = (state * 1_103_515_245 + 12_345) & 0xFFFF_FFFF
            frac = state / 0x1_0000_0000
            radius = radii[idx % 8]  # cycle every radius so each gets 256 angles
            theta = frac * 2 * math.pi - math.pi
            y = round(math.sin(theta) * radius)
            x = round(math.cos(theta) * radius)
            r = atan2_mag(y, x)
            self.assertAlmostEqual(r.angle, math.atan2(y / SCALE, x / SCALE), delta=MAX_ANGLE_ERR_ULP / SCALE)
            if not r.saturated:
                self.assertAlmostEqual(r.mag, math.hypot(y / SCALE, x / SCALE), delta=MAX_MAG_ERR_ULP / SCALE)

    def test_core_against_oracle_coarse_full_grid(self):
        vals = (
            INT32_MIN, INT32_MIN + 1, -3 * SCALE - 7, -SCALE - 1, -SCALE,
            -SCALE // 2 - 3, -5, -1, 0, 1, 5, SCALE // 2 + 3, SCALE,
            SCALE + 1, 3 * SCALE + 7, INT32_MAX - 1, INT32_MAX,
        )
        for y in vals:
            for x in vals:
                with self.subTest(y=y, x=x):
                    r = atan2_mag(y, x)
                    self.assertLessEqual(abs(_ae_ulp(r)), MAX_ANGLE_ERR_ULP)
                    # Saturated magnitude is exact at the clamp (gap reported);
                    # unsaturated magnitude must meet the datapath bound.
                    if r.saturated:
                        self.assertEqual(r.mag_q, INT32_MAX)
                        self.assertLessEqual(_me_ulp(r), 1)  # clamp at/over true value
                    else:
                        self.assertLessEqual(abs(_me_ulp(r)), MAX_MAG_ERR_ULP)


class TestQuadrantsAndOffset(unittest.TestCase):
    def test_offset_by_quadrant(self):
        s = SCALE
        self.assertEqual(atan2_mag(s, s).angle_offset_q, 0)       # Q1
        self.assertEqual(atan2_mag(-s, s).angle_offset_q, 0)      # Q4
        self.assertEqual(atan2_mag(s, -s).angle_offset_q, PI_Q)   # Q2 -> +pi
        self.assertEqual(atan2_mag(-s, -s).angle_offset_q, -PI_Q) # Q3 -> -pi
        self.assertEqual(atan2_mag(s, 0).angle_offset_q, 0)
        self.assertEqual(atan2_mag(-s, 0).angle_offset_q, 0)
        self.assertEqual(atan2_mag(0, -s).angle_offset_q, PI_Q)

    def test_quadrant_signs(self):
        r11 = atan2_mag(SCALE, SCALE)
        r1m = atan2_mag(SCALE, -SCALE)
        rm1 = atan2_mag(-SCALE, SCALE)
        rmm = atan2_mag(-SCALE, -SCALE)
        self.assertGreater(r11.angle_q, 0)
        self.assertGreater(r1m.angle_q, 0)
        self.assertLess(rm1.angle_q, 0)
        self.assertLess(rmm.angle_q, 0)
        # Symmetric pair angles are near-negatives (floor shifts allow bias).
        self.assertLessEqual(abs(r11.angle_q + rm1.angle_q), 16)
        self.assertLessEqual(abs(r1m.angle_q + rmm.angle_q), 16)


class TestInt32MinAndSaturation(unittest.TestCase):
    def test_int32_min_coordinates_do_not_wrap(self):
        for y, x in (
            (INT32_MIN, INT32_MIN),
            (INT32_MIN, 0),
            (0, INT32_MIN),
            (INT32_MIN, 1),
            (1, INT32_MIN),
            (INT32_MIN, INT32_MAX),
            (INT32_MAX, INT32_MIN),
        ):
            with self.subTest(y=y, x=x):
                r = atan2_mag(y, x)  # must not raise on -2**31
                self.assertLessEqual(abs(_ae_ulp(r)), MAX_ANGLE_ERR_ULP)
                self.assertEqual(r.mag_q, INT32_MAX)
                self.assertTrue(r.saturated)

    def test_int32_min_angles_exact_quadrants(self):
        self.assertAlmostEqual(atan2_mag(INT32_MIN, 0).angle, -math.pi / 2, delta=48 / SCALE)
        self.assertAlmostEqual(atan2_mag(0, INT32_MIN).angle, math.pi, delta=48 / SCALE)
        self.assertAlmostEqual(atan2_mag(INT32_MIN, INT32_MIN).angle, -3 * math.pi / 4, delta=48 / SCALE)
        self.assertAlmostEqual(atan2_mag(INT32_MAX, INT32_MIN).angle, 3 * math.pi / 4, delta=48 / SCALE)

    def test_saturation_flag_and_value(self):
        # sqrt(2)*7 > 8 - 1ulp: a (7, 7) vector cannot fit Q4.28.
        r = atan2_mag(7 * SCALE, 7 * SCALE)
        self.assertTrue(r.saturated)
        self.assertEqual(r.mag_q, INT32_MAX)
        # Angle remains valid even though magnitude was clamped.
        self.assertLessEqual(abs(_ae_ulp(r)), MAX_ANGLE_ERR_ULP)
        self.assertAlmostEqual(r.angle, math.pi / 4, delta=48 / SCALE)

    def test_just_under_saturation_is_exact_enough(self):
        r = atan2_mag(0, 7 * SCALE)
        self.assertFalse(r.saturated)
        self.assertLessEqual(abs(r.mag_q - 7 * SCALE), MAX_MAG_ERR_ULP)

    def test_saturated_magnitude_error_is_clamp_gap(self):
        r = atan2_mag(INT32_MIN, INT32_MIN)
        # True magnitude ~sqrt(2)*8; clamped output sits well below it.
        self.assertTrue(r.saturated)
        self.assertLess(_me_ulp(r), -1.0)

    def test_boundary_max_axis_documented_edge(self):
        # The unclamped datapath can overshoot the ideal by ~1 ulp at the
        # ceiling and clamp onto the exact representable maximum: the flag
        # reports datapath saturation, the output value is the correct max.
        r = atan2_mag(0, INT32_MAX)
        self.assertEqual(r.mag_q, INT32_MAX)
        self.assertLessEqual(abs(_me_ulp(r)), 1)


class TestRoundsTrace(unittest.TestCase):
    def test_trace_shape(self):
        r = atan2_mag(3 * SCALE, 4 * SCALE)
        self.assertEqual(len(r.rounds), ITERATIONS)
        for i, st in enumerate(r.rounds):
            self.assertEqual(st.index, i)
            self.assertIn(st.direction, (-1, 1))  # nonzero vector never idles
            # z is the angle accumulator and always fits a signed word.
            self.assertTrue(INT32_MIN <= st.z <= INT32_MAX)

    def test_y_is_driven_toward_zero(self):
        r = atan2_mag(SCALE, 2 * SCALE)
        self.assertLessEqual(abs(r.rounds[-1].y), 4)

    def test_zero_vector_trace_direction_zero(self):
        r = atan2_mag(0, 0)
        self.assertTrue(all(st.direction == 0 for st in r.rounds))


class TestThinWrappersAndFloats(unittest.TestCase):
    def test_atan2_wrapper(self):
        self.assertEqual(atan2(SCALE, SCALE), atan2_mag(SCALE, SCALE).angle_q)
        self.assertIsInstance(atan2(1, 2), int)

    def test_magnitude_wrapper(self):
        self.assertEqual(magnitude(3 * SCALE, 4 * SCALE), atan2_mag(3 * SCALE, 4 * SCALE).mag_q)
        self.assertIsInstance(magnitude(1, 2), int)

    def test_float_wrapper_normal(self):
        r = atan2_mag_radians(-0.5, 1.25)
        self.assertIsInstance(r, Atan2Result)
        self.assertAlmostEqual(r.angle, math.atan2(-0.5, 1.25), delta=MAX_ANGLE_ERR_ULP / SCALE)

    def test_float_wrapper_rejects_nonfinite(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(NonFiniteValueError):
                atan2_mag_radians(bad, 1.0)
            with self.assertRaises(NonFiniteValueError):
                atan2_mag_radians(1.0, bad)

    def test_float_wrapper_rejects_out_of_range(self):
        with self.assertRaises(FixedPointOverflowError):
            atan2_mag_radians(8.0, 0.0)
        with self.assertRaises(FixedPointOverflowError):
            atan2_mag_radians(0.0, -8.5)


class TestInputValidation(unittest.TestCase):
    def test_bool_rejected(self):
        for bad in (True, False):
            with self.assertRaises(InvalidArgumentTypeError):
                atan2_mag(bad, 1)
            with self.assertRaises(InvalidArgumentTypeError):
                atan2_mag(1, bad)

    def test_non_int_rejected(self):
        for bad in (1.5, "1", None, [1], 1 + 0j):
            with self.assertRaises(InvalidArgumentTypeError, msg=f"{bad!r} as y"):
                atan2_mag(bad, 1)
            with self.assertRaises(InvalidArgumentTypeError, msg=f"{bad!r} as x"):
                atan2_mag(1, bad)

    def test_out_of_range_rejected(self):
        with self.assertRaises(FixedPointOverflowError) as cm:
            atan2_mag(INT32_MAX + 1, 0)
        self.assertEqual(cm.exception.where, "y_q")
        with self.assertRaises(FixedPointOverflowError) as cm:
            atan2_mag(0, INT32_MIN - 1)
        self.assertEqual(cm.exception.where, "x_q")

    def test_all_errors_share_base(self):
        for exc_type in (
            FixedPointOverflowError,
            InvalidArgumentTypeError,
            NonFiniteValueError,
        ):
            self.assertTrue(issubclass(exc_type, CordicError))

    def test_no_partial_state_on_failure(self):
        good = atan2_mag(SCALE, SCALE)
        with self.assertRaises(CordicError):
            atan2_mag(True, 1)
        with self.assertRaises(CordicError):
            atan2_mag_radians(1.0, math.inf)
        with self.assertRaises(CordicError):
            atan2_mag(INT32_MIN - 1, 0)
        self.assertEqual(good, atan2_mag(SCALE, SCALE))

    def test_oracle_validates_too(self):
        with self.assertRaises(CordicError):
            atan2_mag_ref(True, 0)
        with self.assertRaises(CordicError):
            atan2_mag_ref(0, INT32_MAX + 1)


class TestNewPrimitives(unittest.TestCase):
    def test_mul_q_basic(self):
        self.assertEqual(mul_q(SCALE, SCALE), SCALE)          # 1 * 1 = 1
        self.assertEqual(mul_q(0, INT32_MAX), 0)
        self.assertEqual(mul_q(-SCALE, SCALE), -SCALE)       # -1 * 1 = -1
        self.assertEqual(mul_q(3, 0), 0)

    def test_mul_q_ties_to_even(self):
        half_ulp = 1 << (28 - 1)  # 2**27: a product of exactly this is a 0.5 tie
        self.assertEqual(mul_q(1, half_ulp), 0)   # 0.5 ulp -> 0 (even)
        self.assertEqual(mul_q(3, half_ulp), 2)   # 1.5 ulp -> 2 (even)
        self.assertEqual(mul_q(5, half_ulp), 2)   # 2.5 ulp -> 2 (even)

    def test_mul_q_rejects_bool(self):
        with self.assertRaises(InvalidArgumentTypeError):
            mul_q(True, 1)

    def test_rshift_even_ties(self):
        self.assertEqual(rshift_even(3, 1), 2)   # 1.5 -> 2 (even)
        self.assertEqual(rshift_even(1, 1), 0)   # 0.5 -> 0 (even)
        self.assertEqual(rshift_even(5, 1), 2)   # 2.5 -> 2 (even)
        self.assertEqual(rshift_even(-3, 1), -2) # -1.5 -> -2 (even)
        self.assertEqual(rshift_even(7, 0), 7)

    def test_rshift_rejects_negative_shift(self):
        with self.assertRaises(InvalidArgumentTypeError):
            rshift_even(1, -1)

    def test_saturate_int32(self):
        self.assertEqual(saturate_int32(0), (0, False))
        self.assertEqual(saturate_int32(INT32_MAX), (INT32_MAX, False))
        self.assertEqual(saturate_int32(INT32_MAX + 1), (INT32_MAX, True))
        self.assertEqual(saturate_int32(INT32_MIN), (INT32_MIN, False))
        self.assertEqual(saturate_int32(INT32_MIN - 1), (INT32_MIN, True))

    def test_frac_bits_unchanged(self):
        self.assertEqual(FRAC_BITS, 28)


if __name__ == "__main__":
    unittest.main()

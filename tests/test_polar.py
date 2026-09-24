"""Vectoring-mode CORDIC tests: atan2 + magnitude on axes, quadrants, edges.

Expectations never come from the implementation under test: they are
hand-computed analytic values (unit axes, 45-degree diagonals, a 3-4-5
triangle), exact integer identities (magnitude^2 = x^2 + y^2 reconstructed),
and the independent Decimal oracle ``polar_ref`` (Decimal atan series +
Decimal square root, no CORDIC code path).
"""

import math
import unittest
from decimal import Decimal

import cordic_q428.cordic as _core

from cordic_q428 import (
    GAIN_INV_Q,
    GAIN_PREFIX_INV_Q,
    HALF_PI_Q,
    INT32_MAX,
    INT32_MIN,
    ITERATIONS,
    PI_Q,
    SCALE,
    CordicError,
    FixedPointOverflowError,
    InvalidArgumentTypeError,
    NonFiniteValueError,
    PolarResult,
    polar,
    polar_ref,
    polar_xy,
)

# Non-saturating outputs stay within a generous analytic bound.  Shift
# truncation contributes <= 1 count per round amplified by the ~1.647 gain,
# plus <= 1 ulp table/gain quantization each; 80 ulp matches the rotation
# mode's contract.  Observed errors are far smaller (a few ulp for r >= 1).
MAX_ERR_ULP = 80


def _err_ulp(result: PolarResult) -> tuple[Decimal, Decimal]:
    return result.angle_error * SCALE, result.mag_error * SCALE


class TestAxes(unittest.TestCase):
    """The four coordinate axes are analytic, hand-computed results."""

    def test_positive_x_axis(self):
        r = polar(SCALE, 0)
        self.assertEqual(r.mag_q, SCALE)       # exactly 1.0, no noise
        self.assertEqual(r.angle_q, 0)         # exactly 0
        self.assertEqual(r.angle_offset_q, 0)
        self.assertFalse(r.magnitude_saturated)

    def test_positive_y_axis(self):
        r = polar(0, SCALE)
        self.assertLessEqual(abs(r.mag_q - SCALE), 1)
        self.assertLessEqual(abs(r.angle_q - HALF_PI_Q), 1)

    def test_negative_y_axis(self):
        r = polar(0, -SCALE)
        self.assertLessEqual(abs(r.mag_q - SCALE), 1)
        self.assertLessEqual(abs(r.angle_q + HALF_PI_Q), 1)

    def test_negative_x_axis_is_plus_pi(self):
        # No signed integer zero: atan2(0, negative) is defined as +pi.
        r = polar(-SCALE, 0)
        self.assertEqual(r.angle_offset_q, PI_Q)
        self.assertLessEqual(abs(r.angle_q - PI_Q), 1)
        self.assertLessEqual(abs(r.mag_q - SCALE), 1)

    def test_axis_radius_scales_linearly(self):
        # Magnitude on the +x axis must equal the input coordinate exactly at
        # several radii (fold leaves (r, 0) untouched, zero micro-rotations).
        for rq in (1, 2, SCALE, SCALE * 3, INT32_MAX):
            with self.subTest(rq=rq):
                res = polar(rq, 0)
                self.assertEqual(res.mag_q, rq)
                self.assertEqual(res.angle_q, 0)


class TestQuadrants(unittest.TestCase):
    """All four quadrants on the unit diagonals and a 3-4-5 triangle."""

    def test_diagonal_45_degrees(self):
        # (1, 1): angle pi/4 exactly, magnitude sqrt(2).
        r = polar(SCALE, SCALE)
        self.assertEqual(r.angle_q, PI_Q // 4)  # folded by exactly one round
        self.assertFalse(r.magnitude_saturated)
        # sqrt(2) within a couple of ulp.
        self.assertLessEqual(abs(r.mag_q / SCALE - math.sqrt(2)), 2 / SCALE + 1e-12)

    def test_all_four_quadrants_diagonal(self):
        expected_angle = (
            PI_Q // 4,            # QI
            PI_Q - PI_Q // 4,     # QII  (3*pi/4)
            -(PI_Q - PI_Q // 4),  # QIII (-3*pi/4)
            -(PI_Q // 4),         # QIV
        )
        for (x, y), want in zip(
            ((SCALE, SCALE), (-SCALE, SCALE), (-SCALE, -SCALE), (SCALE, -SCALE)),
            expected_angle,
        ):
            with self.subTest(x=x, y=y):
                r = polar(x, y)
                self.assertLessEqual(abs(r.angle_q - want), 2)
                self.assertLessEqual(abs(r.mag_q / SCALE - math.sqrt(2)), 2 / SCALE + 1e-12)

    def test_345_triangle(self):
        # (3, 4): magnitude exactly 5; angle atan2(4, 3) vs Decimal oracle.
        r = polar(3 * SCALE, 4 * SCALE)
        self.assertLessEqual(abs(r.mag_q - 5 * SCALE), MAX_ERR_ULP)
        ae, _ = _err_ulp(r)
        self.assertLessEqual(abs(ae), MAX_ERR_ULP)

    def test_quadrant_offsets_recorded(self):
        # Dominant-axis fold: equal-magnitude diagonals are owned by the x
        # component, so the left-half-plane diagonals fold by a half turn.
        offs = {
            (SCALE, SCALE): 0,
            (-SCALE, SCALE): PI_Q,
            (-SCALE, -SCALE): -PI_Q,
            (SCALE, -SCALE): 0,
            (-SCALE, 0): PI_Q,
            (0, SCALE): HALF_PI_Q,
            (0, -SCALE): -HALF_PI_Q,
        }
        for (x, y), want in offs.items():
            with self.subTest(x=x, y=y):
                self.assertEqual(polar(x, y).angle_offset_q, want)


class TestZeroVector(unittest.TestCase):
    def test_zero_vector_defined_result(self):
        r = polar(0, 0)
        self.assertEqual(r.mag_q, 0)
        self.assertEqual(r.angle_q, 0)
        self.assertEqual(r.angle_offset_q, 0)
        self.assertFalse(r.magnitude_saturated)
        # No micro-rotation is performed on the degenerate vector.
        self.assertTrue(all(st.direction == 0 for st in r.rounds))

    def test_zero_matches_oracle(self):
        a, m = polar_ref(0, 0)
        self.assertEqual(a, 0)
        self.assertEqual(m, 0)

    def test_rounds_length_unchanged(self):
        self.assertEqual(len(polar(0, 0).rounds), ITERATIONS)


class TestSaturation(unittest.TestCase):
    def test_diagonal_corner_saturates(self):
        # (-2**31, -2**31): true magnitude sqrt(2)*8 > the 8.0 cap.
        r = polar(INT32_MIN, INT32_MIN)
        self.assertTrue(r.magnitude_saturated)
        self.assertEqual(r.mag_q, INT32_MAX)
        # Angle is still valid even when magnitude saturated.
        self.assertLessEqual(abs(r.angle_q + (PI_Q - PI_Q // 4)), 2)

    def test_negative_x_axis_radius_8_saturates(self):
        # Magnitude exactly 8.0 exceeds the largest representable 8.0-ulp.
        r = polar(INT32_MIN, 0)
        self.assertTrue(r.magnitude_saturated)
        self.assertEqual(r.mag_q, INT32_MAX)

    def test_max_corner_saturates(self):
        r = polar(INT32_MAX, INT32_MAX)
        self.assertTrue(r.magnitude_saturated)
        self.assertEqual(r.mag_q, INT32_MAX)

    def test_max_x_axis_does_not_saturate(self):
        r = polar(INT32_MAX, 0)
        self.assertFalse(r.magnitude_saturated)
        self.assertEqual(r.mag_q, INT32_MAX)

    def test_saturation_predicate_boundary(self):
        # Largest y with x^2+y^2 <= INT32_MAX^2: not saturated; y+1 is.
        x = INT32_MAX - 100
        y_edge = math.isqrt(INT32_MAX * INT32_MAX - x * x)
        self.assertFalse(polar(x, y_edge).magnitude_saturated)
        self.assertTrue(polar(x, y_edge + 1).magnitude_saturated)
        self.assertEqual(polar(x, y_edge + 1).mag_q, INT32_MAX)

    def test_saturated_magnitude_error_is_honest(self):
        # The reported error must equal the saturation gap (output cap minus
        # the true magnitude), i.e. it honestly reflects how far the clamped
        # output is below the reference rather than pretending to match.
        r = polar(INT32_MIN, INT32_MIN)
        _, ref_mag = polar_ref(INT32_MIN, INT32_MIN)
        self.assertLess(r.mag_error, Decimal(0))
        self.assertEqual(r.mag_error, Decimal(INT32_MAX) / SCALE - ref_mag)


class TestAgainstOracle(unittest.TestCase):
    """Vectoring core vs the independent Decimal atan/sqrt oracle."""

    def test_radius_angle_grid(self):
        worst_a = worst_m = Decimal(0)
        for ri in range(1, 13):
            rq = ri * (INT32_MAX // 13)
            for k in range(64):
                th = k * (2 * math.pi / 64)
                x = max(INT32_MIN, min(INT32_MAX, int(rq * math.cos(th))))
                y = max(INT32_MIN, min(INT32_MAX, int(rq * math.sin(th))))
                with self.subTest(r=rq, k=k):
                    r = polar(x, y)
                    if r.magnitude_saturated:
                        continue
                    ae, me = _err_ulp(r)
                    worst_a = max(worst_a, abs(ae))
                    worst_m = max(worst_m, abs(me))
                    self.assertLessEqual(abs(ae), MAX_ERR_ULP)
                    self.assertLessEqual(abs(me), MAX_ERR_ULP)
        self.assertGreater(worst_a, 0)
        self.assertGreater(worst_m, 0)

    def test_deterministic_enum_r_ge_one(self):
        # Deterministic pseudo-enumeration; angle comparison is meaningful only
        # once the vector magnitude is at least ~1 real (coordinates well above
        # the ulp floor).  Magnitude is checked at every non-saturated point.
        max_a = max_m = 0.0
        for i in range(1500):
            x = ((i * 2654435761) % (2 ** 32)) - 2 ** 31
            y = ((i * 40503 + 12345) % (2 ** 32)) - 2 ** 31
            x = max(INT32_MIN, min(INT32_MAX, x))
            y = max(INT32_MIN, min(INT32_MAX, y))
            r = polar(x, y)
            if r.magnitude_saturated:
                continue
            ae = abs(float(r.angle_error)) * SCALE
            me = abs(float(r.mag_error)) * SCALE
            max_m = max(max_m, me)
            if r.mag_q >= SCALE:
                max_a = max(max_a, ae)
            self.assertLessEqual(me, MAX_ERR_ULP, f"mag x={x} y={y}")
        self.assertLessEqual(max_a, MAX_ERR_ULP)


class TestMagnitudeIdentity(unittest.TestCase):
    def test_pythagorean_reconstruction(self):
        # Reconstruct x^2+y^2 from the reported magnitude (within error) for a
        # spread of non-saturated vectors, including the most-negative corner.
        for (x, y) in (
            (SCALE, 0), (0, SCALE), (3 * SCALE, 4 * SCALE),
            (123456789, 999999937), (INT32_MAX, 0),
            (INT32_MIN, 1), (-123456789, -777777777),
        ):
            with self.subTest(x=x, y=y):
                r = polar(x, y)
                if r.magnitude_saturated:
                    continue
                true_r = math.sqrt(x * x + y * y) / SCALE
                self.assertAlmostEqual(r.mag, true_r, delta=MAX_ERR_ULP / SCALE + 1e-9)


class TestInputValidation(unittest.TestCase):
    def test_bool_rejected(self):
        for bad in (True, False):
            with self.assertRaises(InvalidArgumentTypeError):
                polar(bad, 0)
            with self.assertRaises(InvalidArgumentTypeError):
                polar(0, bad)

    def test_non_int_rejected(self):
        for bad in (1.5, "1", None, [1], 1 + 0j):
            with self.assertRaises(InvalidArgumentTypeError, msg=f"{bad!r} accepted"):
                polar(bad, 0)
            with self.assertRaises(InvalidArgumentTypeError):
                polar(0, bad)

    def test_out_of_range_rejected(self):
        for bad in (INT32_MAX + 1, INT32_MIN - 1, 2 ** 40):
            with self.assertRaises(FixedPointOverflowError):
                polar(bad, 0)
            with self.assertRaises(FixedPointOverflowError):
                polar(0, bad)

    def test_error_locatable(self):
        try:
            polar(INT32_MAX + 1, 0)
        except FixedPointOverflowError as exc:
            self.assertEqual(exc.where, "x_q")
            self.assertEqual(exc.value, INT32_MAX + 1)
        else:
            self.fail("expected FixedPointOverflowError")
        try:
            polar(0, INT32_MIN - 1)
        except FixedPointOverflowError as exc:
            self.assertEqual(exc.where, "y_q")
        else:
            self.fail("expected FixedPointOverflowError")

    def test_float_wrapper_rejects_nan_inf_and_range(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(NonFiniteValueError):
                polar_xy(bad, 0.0)
            with self.assertRaises(NonFiniteValueError):
                polar_xy(0.0, bad)
        with self.assertRaises(FixedPointOverflowError):
            polar_xy(8.5, 0.0)
        with self.assertRaises(FixedPointOverflowError):
            polar_xy(0.0, -8.5)

    def test_float_wrapper_normal(self):
        r = polar_xy(3.0, 4.0)
        self.assertIsInstance(r, PolarResult)
        self.assertAlmostEqual(r.mag, 5.0, delta=80 / SCALE)

    def test_no_partial_state_on_failure(self):
        good = polar(SCALE, SCALE)
        with self.assertRaises(CordicError):
            polar(True, 0)
        with self.assertRaises(CordicError):
            polar(0, INT32_MAX + 1)
        with self.assertRaises(CordicError):
            polar_xy(math.inf, 0.0)
        self.assertEqual(good, polar(SCALE, SCALE))


class TestResultStructure(unittest.TestCase):
    def test_rounds_trace(self):
        r = polar(2 * SCALE, SCALE)
        self.assertEqual(len(r.rounds), ITERATIONS)
        bound = _core._VECTOR_REG_BOUND
        for i, st in enumerate(r.rounds):
            self.assertEqual(st.index, i)
            self.assertIn(st.direction, (1, -1, 0))
            # z stays in strict int32; x/y stay in the documented wide envelope.
            self.assertTrue(INT32_MIN <= st.z <= INT32_MAX)
            self.assertTrue(-bound <= st.x <= bound, f"round {i} x={st.x}")
            self.assertTrue(-bound <= st.y <= bound, f"round {i} y={st.y}")
        # Once a no-op (direction 0) appears, all later rounds are no-ops.
        dirs = [st.direction for st in r.rounds]
        if 0 in dirs:
            first_zero = dirs.index(0)
            self.assertTrue(all(d == 0 for d in dirs[first_zero:]))

    def test_worst_corner_registers_within_envelope(self):
        # The most-negative corner drives the largest intermediate registers.
        for (x, y) in (
            (INT32_MIN, INT32_MIN), (INT32_MIN, INT32_MAX),
            (INT32_MAX, INT32_MIN), (INT32_MIN, 0),
        ):
            r = polar(x, y)
            for st in r.rounds:
                self.assertTrue(-_core._VECTOR_REG_BOUND <= st.x <= _core._VECTOR_REG_BOUND)
                self.assertTrue(-_core._VECTOR_REG_BOUND <= st.y <= _core._VECTOR_REG_BOUND)

    def test_errors_are_decimal(self):
        r = polar(SCALE, SCALE)
        self.assertIsInstance(r.angle_error, Decimal)
        self.assertIsInstance(r.mag_error, Decimal)

    def test_float_views(self):
        r = polar(SCALE, SCALE)
        self.assertAlmostEqual(r.mag, r.mag_q / SCALE)
        self.assertAlmostEqual(r.angle, r.angle_q / SCALE)

    def test_prefix_gain_table(self):
        self.assertEqual(len(GAIN_PREFIX_INV_Q), ITERATIONS + 1)
        self.assertEqual(GAIN_PREFIX_INV_Q[0], SCALE)
        self.assertEqual(GAIN_PREFIX_INV_Q[ITERATIONS], GAIN_INV_Q)


if __name__ == "__main__":
    unittest.main()

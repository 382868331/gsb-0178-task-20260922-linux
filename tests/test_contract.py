"""Contract tests: input validation, error semantics, shift/quantization rules."""

import math
import unittest

from cordic_q428 import (
    ATAN_TABLE_Q,
    FRAC_BITS,
    GAIN_INV_Q,
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
    float_to_q,
    mul_q,
    q_to_float,
    sar,
    sin_cos,
    sin_cos_radians,
)


class TestInputValidation(unittest.TestCase):
    def test_bool_rejected_as_angle(self):
        for bad in (True, False):
            with self.assertRaises(InvalidArgumentTypeError, msg=f"bool {bad} accepted"):
                sin_cos(bad)

    def test_non_int_rejected(self):
        for bad in (1.5, "1.0", None, [1], 1 + 0j):
            with self.assertRaises(InvalidArgumentTypeError, msg=f"{bad!r} accepted"):
                sin_cos(bad)

    def test_float_nan_inf_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(NonFiniteValueError, msg=f"{bad} accepted"):
                sin_cos_radians(bad)
            with self.assertRaises(NonFiniteValueError):
                float_to_q(bad)

    def test_out_of_range_angle_rejected(self):
        # Q4.28 covers [-8.0, 8.0); 8.0 exactly is not representable.
        with self.assertRaises(FixedPointOverflowError):
            sin_cos_radians(8.0)
        with self.assertRaises(FixedPointOverflowError):
            sin_cos_radians(-8.5)
        # Integer path: outside signed 32-bit.
        with self.assertRaises(FixedPointOverflowError):
            sin_cos(INT32_MAX + 1)
        with self.assertRaises(FixedPointOverflowError):
            sin_cos(INT32_MIN - 1)

    def test_error_is_locatable(self):
        try:
            sin_cos(INT32_MAX + 1)
        except FixedPointOverflowError as exc:
            self.assertEqual(exc.where, "angle_q")
            self.assertEqual(exc.value, INT32_MAX + 1)
        else:
            self.fail("expected FixedPointOverflowError")

    def test_all_errors_share_base(self):
        for exc_type in (
            FixedPointOverflowError,
            InvalidArgumentTypeError,
            NonFiniteValueError,
        ):
            self.assertTrue(issubclass(exc_type, CordicError))

    def test_no_partial_state_on_failure(self):
        # sin_cos is pure: a rejected call must not disturb later calls.
        good_before = sin_cos(0)
        with self.assertRaises(CordicError):
            sin_cos(True)
        with self.assertRaises(CordicError):
            sin_cos_radians(math.inf)
        self.assertEqual(good_before, sin_cos(0))


class TestShiftAndQuantization(unittest.TestCase):
    def test_sar_rounds_toward_negative_infinity(self):
        self.assertEqual(sar(-3, 1), -2)  # floor(-1.5) = -2, not -1
        self.assertEqual(sar(-1, 1), -1)
        self.assertEqual(sar(3, 1), 1)
        self.assertEqual(sar(-1, 28), -1)  # any negative shifts to -1 eventually
        self.assertEqual(sar(1, 28), 0)

    def test_sar_rejects_negative_shift(self):
        with self.assertRaises(InvalidArgumentTypeError):
            sar(1, -1)

    def test_float_to_q_ties_to_even(self):
        # 0.5 ulp ties: 0.5/SCALE -> 0 (even), 1.5/SCALE -> 2 (even).
        self.assertEqual(float_to_q(0.5 / SCALE), 0)
        self.assertEqual(float_to_q(1.5 / SCALE), 2)
        self.assertEqual(float_to_q(2.5 / SCALE), 2)

    def test_roundtrip_q_float(self):
        for q in (0, 1, -1, INT32_MAX, INT32_MIN, 123456789):
            self.assertEqual(float_to_q(q_to_float(q)), q)

    def test_mul_q_basic(self):
        # Q4.28 product: (a * b) >> 28 with ties-to-even.
        self.assertEqual(mul_q(SCALE, SCALE), SCALE)          # 1.0 * 1.0 = 1.0
        self.assertEqual(mul_q(SCALE, 0), 0)
        self.assertEqual(mul_q(2 * SCALE, 3 * SCALE), 6 * SCALE)
        # Sign handling.
        self.assertEqual(mul_q(SCALE, -SCALE), -SCALE)
        self.assertEqual(mul_q(-SCALE, -SCALE), SCALE)

    def test_mul_q_ties_to_even(self):
        # Build an exact 0.5-ulp tie: (SCALE+1)/2 * 2 = SCALE+1, so multiply
        # (SCALE+1) by 2*... easier: a*b with fractional part exactly 0.5.
        # (SCALE + 1) * (SCALE // 2) has remainder SCALE//2 exactly when the
        # +1 term contributes 0.5 ulp; check the even quotient is chosen.
        half = SCALE // 2
        # (SCALE+1)*half = half*SCALE + half -> quotient half, remainder half
        # (an exact tie): half=2**27 is even, so it stays half (round to even).
        self.assertEqual(mul_q(SCALE + 1, half), half)
        # Odd quotient tie: (SCALE+1)*(half+1) -> quotient half+1, remainder
        # half + 1 ... verify sign-correctness on a clean negative tie instead.
        # -(SCALE+1) * half: Python divmod keeps remainder positive; quotient is
        # -(half+1), remainder half -> tie rounds to the even quotient -half.
        self.assertEqual(mul_q(-(SCALE + 1), half), -half)

    def test_mul_q_rejects_bool_and_non_int(self):
        for bad in (True, False, 1.0, "1"):
            with self.assertRaises(InvalidArgumentTypeError):
                mul_q(bad, SCALE)
            with self.assertRaises(InvalidArgumentTypeError):
                mul_q(SCALE, bad)

    def test_table_is_ties_to_even_quantized(self):
        # atan(2**-29) * 2**28 = 0.5 exactly-ish; ties-to-even gives 0.
        self.assertEqual(ATAN_TABLE_Q[29], 0)
        self.assertEqual(ATAN_TABLE_Q[28], 1)
        # atan(1) = pi/4 -> well-known Q4.28 value.
        self.assertEqual(ATAN_TABLE_Q[0], 210828714)  # round(pi/4 * 2**28)

    def test_gain_inverse_value(self):
        # Known CORDIC inverse gain for 32 iterations ~ 0.6072529350088814;
        # the quantized integer may differ from the true value by <= 0.5 ulp.
        self.assertLessEqual(abs(GAIN_INV_Q / SCALE - 0.6072529350088814), 0.5 / SCALE + 1e-12)

    def test_pi_constants_consistent(self):
        self.assertEqual(PI_Q, 843314857)  # round(pi * 2**28)
        self.assertEqual(HALF_PI_Q, 421657428)  # round(pi/2 * 2**28)
        self.assertEqual(FRAC_BITS, 28)
        self.assertEqual(ITERATIONS, 32)


if __name__ == "__main__":
    unittest.main()

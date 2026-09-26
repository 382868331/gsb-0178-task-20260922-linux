"""Tests for the hyperbolic rotation-mode CORDIC (cosh/sinh, |z| <= 1).

Covers: zero, positive/negative boundaries, configurations around the
repeated iterations (i = 4, 13, 40), per-config measured max error against
the independent Decimal exponential reference, and rejection semantics.
"""

import math
import unittest
from decimal import Decimal

from cordic_q428 import (
    ATANH_TABLE_Q,
    GAIN_HYP_INV_Q,
    HYPER_ITERATIONS,
    HYPER_MAX_Z_Q,
    HYPER_REPEAT_INDICES,
    INT32_MAX,
    INT32_MIN,
    SCALE,
    CordicError,
    FixedPointOverflowError,
    InputRangeError,
    InvalidArgumentTypeError,
    NonFiniteValueError,
    sinh_cosh,
    sinh_cosh_radians,
    sinh_cosh_ref,
)

# Analytic bound: <= 1 ulp gain quantization + 43 shift truncations of <= 1
# ulp each amplified by the ~1.21 hyperbolic gain + <= 1 ulp table
# quantization on z, with margin.  Observed errors are far smaller (~22 ulp).
MAX_ERR_ULP = 96


def _err_ulp(result):
    return result.sinh_error * SCALE, result.cosh_error * SCALE


class TestHyperInputValidation(unittest.TestCase):
    def test_bool_rejected(self):
        for bad in (True, False):
            with self.assertRaises(InvalidArgumentTypeError):
                sinh_cosh(bad)

    def test_non_int_rejected(self):
        for bad in (0.5, "0", None, [0]):
            with self.assertRaises(InvalidArgumentTypeError):
                sinh_cosh(bad)

    def test_nan_inf_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(NonFiniteValueError):
                sinh_cosh_radians(bad)

    def test_out_of_convergence_range_rejected(self):
        # |z| > 1.0 must be refused, never saturated into a plausible value.
        for q in (SCALE + 1, -SCALE - 1, 2 * SCALE, INT32_MAX, INT32_MIN):
            with self.assertRaises(InputRangeError, msg=f"z_q={q} accepted"):
                sinh_cosh(q)
        with self.assertRaises(InputRangeError):
            sinh_cosh_radians(1.5)

    def test_out_of_int32_rejected(self):
        with self.assertRaises(FixedPointOverflowError):
            sinh_cosh(INT32_MAX + 1)
        with self.assertRaises(FixedPointOverflowError):
            sinh_cosh(INT32_MIN - 1)

    def test_range_error_is_locatable(self):
        try:
            sinh_cosh(SCALE + 1)
        except InputRangeError as exc:
            self.assertEqual(exc.where, "z_q")
            self.assertEqual(exc.value, SCALE + 1)
        else:
            self.fail("expected InputRangeError")

    def test_errors_share_base(self):
        self.assertTrue(issubclass(InputRangeError, CordicError))

    def test_no_partial_state_on_failure(self):
        good = sinh_cosh(0)
        with self.assertRaises(CordicError):
            sinh_cosh(SCALE + 1)
        self.assertEqual(good, sinh_cosh(0))


class TestHyperKnownPoints(unittest.TestCase):
    def test_zero(self):
        r = sinh_cosh(0)
        self.assertLessEqual(abs(r.sinh_q), MAX_ERR_ULP)
        self.assertLessEqual(abs(r.cosh_q - SCALE), MAX_ERR_ULP)

    def test_positive_boundary(self):
        r = sinh_cosh(HYPER_MAX_Z_Q)  # z = 1.0 exactly
        self.assertLessEqual(abs(_err_ulp(r)[0]), MAX_ERR_ULP)
        self.assertLessEqual(abs(_err_ulp(r)[1]), MAX_ERR_ULP)
        self.assertAlmostEqual(r.sinh, 1.1752011936438014, places=7)
        self.assertAlmostEqual(r.cosh, 1.5430806348152437, places=7)

    def test_negative_boundary(self):
        r = sinh_cosh(-HYPER_MAX_Z_Q)  # z = -1.0 exactly
        self.assertLessEqual(abs(_err_ulp(r)[0]), MAX_ERR_ULP)
        self.assertLessEqual(abs(_err_ulp(r)[1]), MAX_ERR_ULP)

    def test_boundary_minus_ulp(self):
        for q in (SCALE - 1, -SCALE + 1, 1, -1):
            with self.subTest(q=q):
                r = sinh_cosh(q)
                se, ce = _err_ulp(r)
                self.assertLessEqual(abs(se), MAX_ERR_ULP)
                self.assertLessEqual(abs(ce), MAX_ERR_ULP)

    def test_cosh_minus_sinh_identity(self):
        # cosh^2 - sinh^2 ~ 1 within accumulated fixed-point error.
        for q in (0, 1, -1, SCALE // 2, -SCALE // 2, SCALE, -SCALE):
            r = sinh_cosh(q)
            ident = (r.cosh_q * r.cosh_q - r.sinh_q * r.sinh_q) / (SCALE * SCALE)
            self.assertAlmostEqual(ident, 1.0, delta=1e-6)


class TestHyperRepeatSchedule(unittest.TestCase):
    def test_repeat_indices(self):
        self.assertEqual(HYPER_REPEAT_INDICES, (4, 13, 40))
        self.assertEqual(HYPER_ITERATIONS, 40)

    def test_rounds_include_repeats(self):
        r = sinh_cosh(SCALE // 3)
        self.assertEqual(len(r.rounds), HYPER_ITERATIONS + len(HYPER_REPEAT_INDICES))
        repeated = [s.index for s in r.rounds if s.repeated]
        self.assertEqual(repeated, [4, 13, 40])
        # Every iteration 1..40 appears exactly once as a non-repeated round.
        firsts = [s.index for s in r.rounds if not s.repeated]
        self.assertEqual(firsts, list(range(1, HYPER_ITERATIONS + 1)))

    def test_round_records_shift_and_rounding(self):
        r = sinh_cosh(SCALE // 2)
        for st in r.rounds:
            self.assertEqual(st.shift, st.index)
            self.assertIn(st.direction, (1, -1))
            # The recorded shifted addends must be the actual sar() results.
            # Recompute from the previous round's registers.
        prev_x, prev_y = GAIN_HYP_INV_Q, 0
        for st in r.rounds:
            self.assertEqual(st.x_shift, prev_x >> st.shift)
            self.assertEqual(st.y_shift, prev_y >> st.shift)
            prev_x, prev_y = st.x, st.y

    def test_gain_inverse_accounts_for_repeats(self):
        # Gain without the repeats would be measurably different; the frozen
        # constant must match the product including i = 4, 13, 40 twice.
        from decimal import Decimal, localcontext

        with localcontext() as ctx:
            ctx.prec = 60
            gain = Decimal(1)
            executed = list(range(1, HYPER_ITERATIONS + 1)) + list(HYPER_REPEAT_INDICES)
            for i in executed:
                gain *= (Decimal(1) - Decimal(2) ** (-2 * i)).sqrt()
            expect = int((1 / gain) * SCALE)
        self.assertLessEqual(abs(GAIN_HYP_INV_Q - expect), 1)

    def test_atanh_table_values(self):
        # atanh(0.5) = 0.5493061443340548... -> Q4.28
        self.assertEqual(ATANH_TABLE_Q[0], 147453245)  # round(atanh(1/2)*2**28)
        # Monotone non-increasing, quantized to 0 well before i = 40.
        self.assertTrue(all(
            ATANH_TABLE_Q[i] >= ATANH_TABLE_Q[i + 1]
            for i in range(HYPER_ITERATIONS - 1)
        ))
        self.assertEqual(ATANH_TABLE_Q[-1], 0)


class TestHyperMeasuredErrorByConfig(unittest.TestCase):
    """Measured max error per configuration group vs the Decimal exp oracle."""

    GROUPS = {
        "zero_and_tiny": [0, 1, -1, 2, -2, 7, -7],
        "positive_boundary": [SCALE - 2, SCALE - 1, SCALE],
        "negative_boundary": [-SCALE + 2, -SCALE + 1, -SCALE],
        # Configurations whose rotation paths exercise the repeated rounds
        # heavily: angles near partial sums of atanh(2**-i) up to i = 4/13.
        "near_repeat_configs": [
            sum(ATANH_TABLE_Q[:4]),           # ~ sum atanh(2**-i), i=1..4
            sum(ATANH_TABLE_Q[:4]) + 1,
            -sum(ATANH_TABLE_Q[:4]),
            -sum(ATANH_TABLE_Q[:4]) - 1,
            SCALE - 3,                        # exercises the full i<=13 prefix
            -SCALE + 3,
        ],
        "mid_range_sweep": [
            -SCALE + k * (2 * SCALE) // 200 for k in range(201)
        ],
    }

    def test_measured_max_error_per_config(self):
        for name, qs in self.GROUPS.items():
            worst = Decimal(0)
            for q in qs:
                r = sinh_cosh(q)
                se, ce = _err_ulp(r)
                worst = max(worst, abs(se), abs(ce))
                self.assertLessEqual(abs(se), MAX_ERR_ULP, f"{name} q={q}")
                self.assertLessEqual(abs(ce), MAX_ERR_ULP, f"{name} q={q}")
            # Errors are real (non-vacuous) except possibly the trivial group.
            print(f"  measured max |error| [{name}] = {float(worst):.3f} ulp")
            self.assertGreater(worst, 0)

    def test_reference_is_decimal_exp_definition(self):
        # Spot-check the oracle against the exponential definition directly.
        s, c = sinh_cosh_ref(SCALE // 2)
        e = math.exp(0.5)
        self.assertAlmostEqual(float(s), (e - 1 / e) / 2, places=15)
        self.assertAlmostEqual(float(c), (e + 1 / e) / 2, places=15)

    def test_reference_rejects_out_of_int32(self):
        with self.assertRaises(InvalidArgumentTypeError):
            sinh_cosh_ref(INT32_MAX + 1)


if __name__ == "__main__":
    unittest.main()

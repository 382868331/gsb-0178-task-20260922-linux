"""Tolerance-driven early termination with provable fixed-point bounds."""

import math
import unittest
from fractions import Fraction

from cordic_q428 import (
    HALF_PI_Q,
    INT32_MAX,
    INT32_MIN,
    ITERATIONS,
    SCALE,
    InvalidArgumentTypeError,
    InvalidArgumentValueError,
    NonFiniteValueError,
    ToleranceNotGuaranteedError,
    sin_cos,
    sin_cos_radians,
)

# Representative angles: zero, ulp edges, quadrant edges and word extremes.
ANGLES = (0, 1, -1, 421657428, -421657428, HALF_PI_Q, INT32_MIN, INT32_MAX)


class TestDefaultUnchanged(unittest.TestCase):
    def test_default_runs_all_iterations(self):
        for q in ANGLES:
            with self.subTest(q=q):
                r = sin_cos(q)
                self.assertEqual(r.steps, ITERATIONS)
                self.assertEqual(len(r.rounds), ITERATIONS)
                self.assertFalse(r.early_stopped)
                self.assertIsNone(r.tolerance)
                self.assertIsNone(r.tolerance_met)
                self.assertIsNotNone(r.error_bound_ulp)

    def test_bound_field_is_exact_fraction(self):
        r = sin_cos(1)
        self.assertIsInstance(r.error_bound_ulp, Fraction)
        self.assertIsInstance(r.error_bound, Fraction)
        self.assertEqual(r.error_bound, r.error_bound_ulp / SCALE)


class TestEarlyTermination(unittest.TestCase):
    def test_relaxed_tolerance_stops_early(self):
        stopped_any = False
        for q in ANGLES:
            with self.subTest(q=q):
                r = sin_cos(q, tol_ulp=500)
                self.assertTrue(r.tolerance_met)
                self.assertLessEqual(r.error_bound_ulp, 500)
                self.assertEqual(len(r.rounds), r.steps)
                self.assertEqual(r.early_stopped, r.steps < ITERATIONS)
                stopped_any |= r.early_stopped
        self.assertTrue(stopped_any, "500 ulp should permit early stop somewhere")

    def test_real_tolerance_matches_ulp_tolerance(self):
        r1 = sin_cos(HALF_PI_Q, tol_ulp=300)
        r2 = sin_cos(HALF_PI_Q, tolerance=300 / SCALE)
        r3 = sin_cos(HALF_PI_Q, tolerance=Fraction(300, SCALE))
        self.assertEqual(r1.steps, r2.steps)
        self.assertEqual(r1.steps, r3.steps)
        self.assertEqual(r1.error_bound_ulp, r2.error_bound_ulp)

    def test_proven_bound_covers_actual_error(self):
        # The guarantee must hold against the independent oracle on a dense
        # deterministic sweep, for the early-stopped results.
        for k in range(401):
            q = INT32_MIN + k * (2 ** 32 - 1) // 400
            for tol in (150, 300, 1000):
                with self.subTest(q=q, tol=tol):
                    r = sin_cos(q, tol_ulp=tol)
                    bound = float(r.error_bound_ulp)
                    self.assertLessEqual(abs(float(r.sin_error) * SCALE) + 1e-9, bound)
                    self.assertLessEqual(abs(float(r.cos_error) * SCALE) + 1e-9, bound)

    def test_bound_does_not_use_round_to_round_change(self):
        # From round 29 on the table entries quantize to 0, so late rounds
        # leave z unchanged; a tolerance slightly below the proven bound of
        # that stalled state must still fail (no "change is tiny => done").
        full = sin_cos(0)
        stalled = sin_cos(0, tol_ulp=200)
        self.assertLess(stalled.steps, 29)
        stalled_z_changes = {
            abs(full.rounds[i].z - full.rounds[i - 1].z) for i in range(29, 32)
        }
        self.assertEqual(stalled_z_changes, {0})
        try:
            sin_cos(0, tol_ulp=1)
        except ToleranceNotGuaranteedError as exc:
            guaranteed = float(exc.bound * SCALE)
        else:  # pragma: no cover - one ulp is far below Q4.28 capability
            self.fail("1 ulp tolerance must be refused")
        # The guarantee floor is above 1 ulp even though the last rounds were
        # numerically stationary; tightening it by one ulp must also fail.
        self.assertGreater(guaranteed, 1)
        with self.assertRaises(ToleranceNotGuaranteedError):
            sin_cos(0, tol_ulp=math.floor(guaranteed) - 1)

    def test_impossible_tolerance_fails_without_relaxing(self):
        with self.assertRaises(ToleranceNotGuaranteedError) as cm:
            sin_cos(123456, tol_ulp=1)
        err = cm.exception
        self.assertEqual(err.steps, ITERATIONS)
        self.assertGreater(err.bound * SCALE, 1)
        self.assertIsNotNone(err.result)
        self.assertEqual(err.result.steps, ITERATIONS)
        self.assertFalse(err.result.tolerance_met)

    def test_radians_wrapper_passes_tolerance(self):
        r = sin_cos_radians(1.0, tol_ulp=300)
        self.assertTrue(r.tolerance_met)
        with self.assertRaises(ToleranceNotGuaranteedError):
            sin_cos_radians(1.0, tol_ulp=1)


class TestToleranceValidation(unittest.TestCase):
    def test_non_positive_rejected(self):
        for bad in (0, -1):
            with self.assertRaises(InvalidArgumentValueError):
                sin_cos(0, tol_ulp=bad)
        for bad in (0, -1e-9, -1.0):
            with self.assertRaises(InvalidArgumentValueError):
                sin_cos(0, tolerance=bad)

    def test_bool_rejected(self):
        with self.assertRaises(InvalidArgumentTypeError):
            sin_cos(0, tol_ulp=True)
        with self.assertRaises(InvalidArgumentTypeError):
            sin_cos(0, tolerance=True)

    def test_non_finite_float_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(NonFiniteValueError):
                sin_cos(0, tolerance=bad)

    def test_bad_type_rejected(self):
        with self.assertRaises(InvalidArgumentTypeError):
            sin_cos(0, tolerance="1e-6")
        with self.assertRaises(InvalidArgumentTypeError):
            sin_cos(0, tol_ulp=1.5)

    def test_mutually_exclusive(self):
        with self.assertRaises(InvalidArgumentValueError):
            sin_cos(0, tolerance=1e-6, tol_ulp=100)


if __name__ == "__main__":
    unittest.main()

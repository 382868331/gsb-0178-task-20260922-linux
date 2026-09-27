"""Tests for tolerance-driven early termination and its certified error bound."""

import math
import unittest
from decimal import Decimal

from cordic_q428 import (
    ERROR_BOUNDS_REAL,
    INT32_MAX,
    INT32_MIN,
    ITERATIONS,
    MIN_GUARANTEED_ERROR_REAL,
    MIN_GUARANTEED_ERROR_STEPS,
    SCALE,
    CordicError,
    InvalidArgumentTypeError,
    ToleranceNotGuaranteedError,
    error_bound_real,
    sin_cos,
    sin_cos_radians,
)


class TestDefaultBehaviorUnchanged(unittest.TestCase):
    def test_default_runs_all_rounds(self):
        r = sin_cos(987654321)
        self.assertEqual(r.steps_executed, ITERATIONS)
        self.assertEqual(len(r.rounds), ITERATIONS)
        self.assertIsNone(r.tolerance_met)
        self.assertIsNone(r.requested_tolerance)

    def test_default_matches_full_bound(self):
        r = sin_cos(-123)
        self.assertEqual(r.error_bound, error_bound_real(ITERATIONS))


class TestEarlyTermination(unittest.TestCase):
    def test_stops_before_end_and_certifies(self):
        r = sin_cos_radians(1.0, max_error=1e-6)
        self.assertLess(r.steps_executed, ITERATIONS)
        self.assertEqual(len(r.rounds), r.steps_executed)
        self.assertTrue(r.tolerance_met)
        self.assertLessEqual(r.error_bound, Decimal("1E-6"))
        # Earliest crossing: the previous round's bound was not sufficient.
        self.assertGreater(error_bound_real(r.steps_executed - 1), Decimal("1E-6"))

    def test_guarantee_is_not_based_on_step_delta(self):
        # The certified bound is angle-independent, so the stopping round
        # must be identical for every input with the same tolerance even
        # though the register trajectories (and their round-to-round deltas)
        # differ widely.  A step-delta heuristic stops at trajectory-
        # dependent rounds; it also cannot certify anything because the
        # un-rotated residual angle is invisible to register deltas.
        q_values = [0, 1, 100, 123456789, INT32_MIN, INT32_MAX]
        from cordic_q428 import HALF_PI_Q, PI_Q
        q_values += [HALF_PI_Q, PI_Q, -HALF_PI_Q, HALF_PI_Q + 1]
        results = [sin_cos(q, max_error=Decimal("1E-6")) for q in q_values]
        steps = {r.steps_executed for r in results}
        self.assertEqual(steps, {22})
        for r in results:
            self.assertEqual(r.error_bound, error_bound_real(22))
            self.assertLessEqual(r.error_bound, Decimal("1E-6"))

    def test_actual_error_is_covered_by_certified_bound(self):
        # Deterministic sweep of inputs and tolerances; the reported bound
        # must dominate the independently computed reference error everywhere.
        tolerances = (
            Decimal("1E-3"),
            Decimal("1E-4"),
            Decimal("1E-5"),
            Decimal("3E-6"),
            Decimal("1E-6"),
            MIN_GUARANTEED_ERROR_REAL,
        )
        count = 0
        for k in range(81):
            q = INT32_MIN + k * (2**32 - 1) // 80
            for tol in tolerances:
                with self.subTest(q=q, tol=tol):
                    r = sin_cos(q, max_error=tol)
                    actual = max(abs(r.sin_error), abs(r.cos_error))
                    self.assertLessEqual(actual, r.error_bound)
                    self.assertLessEqual(r.error_bound, tol)
                    count += 1
        self.assertGreater(count, 0)

    def test_bound_covers_quadrant_edges(self):
        from cordic_q428 import HALF_PI_Q, PI_Q

        for edge in (0, HALF_PI_Q, PI_Q, -HALF_PI_Q, -PI_Q, INT32_MIN, INT32_MAX):
            for delta in (-1, 0, 1):
                q = edge + delta
                if not (INT32_MIN <= q <= INT32_MAX):
                    continue
                r = sin_cos(q, max_error=1e-6)
                actual = max(abs(r.sin_error), abs(r.cos_error))
                self.assertLessEqual(actual, r.error_bound, q)

    def test_result_field_consistency(self):
        r = sin_cos(555, max_error=Decimal("0.0001"))
        self.assertEqual(r.requested_tolerance, Decimal("0.0001"))
        self.assertEqual(r.error_bound, error_bound_real(r.steps_executed))
        self.assertEqual(r.rounds[-1].index, r.steps_executed - 1)


class TestImpossibleTolerance(unittest.TestCase):
    def test_too_tight_fails_explicitly(self):
        with self.assertRaises(ToleranceNotGuaranteedError) as cm:
            sin_cos(1000, max_error=1e-9)
        exc = cm.exception
        self.assertEqual(exc.requested_tolerance, Decimal("1E-9"))
        self.assertEqual(exc.best_bound, MIN_GUARANTEED_ERROR_REAL)
        self.assertEqual(exc.steps, MIN_GUARANTEED_ERROR_STEPS)

    def test_no_silent_relaxation(self):
        with self.assertRaises(ToleranceNotGuaranteedError):
            sin_cos(1000, max_error=MIN_GUARANTEED_ERROR_REAL / 2)
        # Exactly at the attainable minimum: succeeds, reports the min bound.
        r = sin_cos(1000, max_error=MIN_GUARANTEED_ERROR_REAL)
        self.assertTrue(r.tolerance_met)
        self.assertEqual(r.error_bound, MIN_GUARANTEED_ERROR_REAL)

    def test_failure_is_pure(self):
        before = sin_cos(424242)
        with self.assertRaises(CordicError):
            sin_cos(424242, max_error=1e-12)
        self.assertEqual(before, sin_cos(424242))

    def test_is_cordic_error(self):
        self.assertTrue(issubclass(ToleranceNotGuaranteedError, CordicError))


class TestToleranceValidation(unittest.TestCase):
    def test_bad_types_and_values(self):
        for bad in (0, -1, -0.0, True, False, "1e-6", [1e-6], object()):
            with self.assertRaises(InvalidArgumentTypeError, msg=repr(bad)):
                sin_cos(1, max_error=bad)
        for bad in (math.nan, math.inf, -math.inf, Decimal("nan"), Decimal("-inf")):
            with self.assertRaises(InvalidArgumentTypeError, msg=repr(bad)):
                sin_cos(1, max_error=bad)

    def test_float_tolerance_is_exact_decimal(self):
        r = sin_cos(1, max_error=0.0000025)  # 2.5e-6 has an exact short repr
        self.assertEqual(r.requested_tolerance, Decimal("2.5E-6"))


class TestBoundTable(unittest.TestCase):
    def test_table_shape_and_positivity(self):
        self.assertEqual(len(ERROR_BOUNDS_REAL), ITERATIONS)
        self.assertTrue(all(b > 0 for b in ERROR_BOUNDS_REAL))

    def test_bound_at_minimum(self):
        self.assertEqual(
            MIN_GUARANTEED_ERROR_REAL,
            min(ERROR_BOUNDS_REAL),
        )
        self.assertEqual(
            error_bound_real(MIN_GUARANTEED_ERROR_STEPS),
            MIN_GUARANTEED_ERROR_REAL,
        )

    def test_bound_argument_range(self):
        for bad in (0, ITERATIONS + 1, -1):
            with self.assertRaises(ValueError):
                error_bound_real(bad)
        with self.assertRaises(TypeError):
            error_bound_real(True)


if __name__ == "__main__":
    unittest.main()

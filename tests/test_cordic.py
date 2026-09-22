"""Core CORDIC tests: quadrant boundaries, min step, negative shifts, full range."""

import unittest
from decimal import Decimal

from cordic_q428 import (
    HALF_PI_Q,
    INT32_MAX,
    INT32_MIN,
    PI_Q,
    SCALE,
    ITERATIONS,
    q_to_float,
    sin_cos,
)

# Maximum |error| the 32-step Q4.28 CORDIC may show vs the Decimal reference.
# Derived analytically: <= 1 ulp gain quantization + 32 shift truncations of
# <= 1 ulp each amplified by the ~1.647 gain + <= 1 ulp table quantization on
# z.  80 ulp is a safe, non-loose bound; observed errors are far smaller.
MAX_ERR_ULP = 80


def _err_ulp(result) -> tuple[Decimal, Decimal]:
    return result.sin_error * SCALE, result.cos_error * SCALE


class TestKnownAngles(unittest.TestCase):
    def test_zero(self):
        r = sin_cos(0)
        # z=0 leaves a residual of a few ulp after 32 rounds; sin(0) is not
        # required to be exactly 0, only within the error bound.
        self.assertLessEqual(abs(r.sin_q), 4)
        self.assertLessEqual(abs(r.cos_q - SCALE), MAX_ERR_ULP)  # cos(0) = 1 within err
        self.assertEqual(r.sign_flip, 1)
        self.assertEqual(r.reduced_q, 0)

    def test_half_pi(self):
        r = sin_cos(HALF_PI_Q)
        self.assertLessEqual(abs(r.sin_q - SCALE), MAX_ERR_ULP)
        self.assertLessEqual(abs(r.cos_q), MAX_ERR_ULP)

    def test_pi(self):
        r = sin_cos(PI_Q)
        self.assertEqual(r.sign_flip, -1)
        self.assertLessEqual(abs(r.sin_q), MAX_ERR_ULP)
        self.assertLessEqual(abs(r.cos_q + SCALE), MAX_ERR_ULP)

    def test_negative_half_pi(self):
        r = sin_cos(-HALF_PI_Q)
        self.assertLessEqual(abs(r.sin_q + SCALE), MAX_ERR_ULP)
        self.assertLessEqual(abs(r.cos_q), MAX_ERR_ULP)

    def test_three_half_pi_via_int(self):
        # 3*pi/2 is representable as an integer sum of quantized constants.
        angle = HALF_PI_Q + PI_Q
        r = sin_cos(angle)
        self.assertLessEqual(abs(r.sin_q + SCALE), MAX_ERR_ULP)
        self.assertLessEqual(abs(r.cos_q), MAX_ERR_ULP)


class TestQuadrantBoundaries(unittest.TestCase):
    """Angles straddling every quadrant edge by exactly one ulp."""

    def test_boundaries(self):
        edges = (0, HALF_PI_Q, PI_Q, HALF_PI_Q + PI_Q, -HALF_PI_Q, -PI_Q)
        for edge in edges:
            for delta in (-1, 0, 1):
                q = edge + delta
                if not (INT32_MIN <= q <= INT32_MAX):
                    continue
                with self.subTest(edge=edge, delta=delta):
                    r = sin_cos(q)
                    se, ce = _err_ulp(r)
                    self.assertLessEqual(abs(se), MAX_ERR_ULP)
                    self.assertLessEqual(abs(ce), MAX_ERR_ULP)
                    self.assertEqual(len(r.rounds), ITERATIONS)

    def test_reduction_stays_in_range(self):
        for q in range(INT32_MIN, INT32_MAX, 4999993):
            r = sin_cos(q)
            self.assertLessEqual(abs(r.reduced_q), HALF_PI_Q)
            self.assertIn(r.sign_flip, (1, -1))


class TestMinAngleStep(unittest.TestCase):
    def test_one_ulp_angle(self):
        r = sin_cos(1)  # angle = 2**-28 rad
        self.assertLessEqual(abs(r.sin_q - 1), MAX_ERR_ULP)
        self.assertLessEqual(abs(r.cos_q - SCALE), MAX_ERR_ULP)

    def test_negative_one_ulp_angle(self):
        r = sin_cos(-1)
        self.assertLessEqual(abs(r.sin_q + 1), MAX_ERR_ULP)
        self.assertLessEqual(abs(r.cos_q - SCALE), MAX_ERR_ULP)

    def test_monotonic_near_zero(self):
        sins = [sin_cos(q).sin_q for q in range(-4, 5)]
        self.assertEqual(sins, sorted(sins))


class TestNegativeShiftPath(unittest.TestCase):
    """Negative reduced angles force d=-1 rounds and negative x>>i shifts."""

    def test_negative_angles_match_reference(self):
        for q in (-1, -1000, -HALF_PI_Q // 3, -HALF_PI_Q, -PI_Q, INT32_MIN):
            with self.subTest(q=q):
                r = sin_cos(q)
                se, ce = _err_ulp(r)
                self.assertLessEqual(abs(se), MAX_ERR_ULP)
                self.assertLessEqual(abs(ce), MAX_ERR_ULP)

    def test_odd_symmetry(self):
        # Arithmetic shifts round toward -inf, so exact odd symmetry is NOT
        # guaranteed: each of the 32 rounds can contribute up to 1 ulp of
        # sign-dependent bias.  Observed worst case is 7 ulp; bound at 16.
        for q in (1, 123456789, HALF_PI_Q // 2, PI_Q - 1):
            rp, rn = sin_cos(q), sin_cos(-q)
            self.assertLessEqual(abs(rp.sin_q + rn.sin_q), 16)
            self.assertLessEqual(abs(rp.cos_q - rn.cos_q), 16)


class TestFullRange(unittest.TestCase):
    def test_endpoints(self):
        for q in (INT32_MIN, INT32_MAX):
            with self.subTest(q=q):
                r = sin_cos(q)
                se, ce = _err_ulp(r)
                self.assertLessEqual(abs(se), MAX_ERR_ULP)
                self.assertLessEqual(abs(ce), MAX_ERR_ULP)

    def test_sweep_error_bound(self):
        # Deterministic sweep across the whole representable range.
        worst = Decimal(0)
        for k in range(401):
            q = INT32_MIN + k * (2**32 - 1) // 400
            r = sin_cos(q)
            se, ce = _err_ulp(r)
            worst = max(worst, abs(se), abs(ce))
            self.assertLessEqual(abs(se), MAX_ERR_ULP, f"q={q}")
            self.assertLessEqual(abs(ce), MAX_ERR_ULP, f"q={q}")
        # Sanity: errors are real but small (not a vacuous test).
        self.assertGreater(worst, 0)

    def test_pythagorean_identity(self):
        # sin^2 + cos^2 ~ 1 within accumulated fixed-point error.
        for q in (0, 1, -1, HALF_PI_Q, PI_Q, INT32_MIN, INT32_MAX, 999999937):
            r = sin_cos(q)
            s2c2 = (r.sin_q * r.sin_q + r.cos_q * r.cos_q) / (SCALE * SCALE)
            self.assertAlmostEqual(s2c2, 1.0, delta=1e-6)


class TestResultStructure(unittest.TestCase):
    def test_rounds_trace(self):
        r = sin_cos(PI_Q // 3)
        self.assertEqual(len(r.rounds), ITERATIONS)
        for i, st in enumerate(r.rounds):
            self.assertEqual(st.index, i)
            self.assertIn(st.direction, (1, -1))
            for v in (st.x, st.y, st.z):
                self.assertTrue(INT32_MIN <= v <= INT32_MAX)
        # z must be driven toward zero by the iteration.
        self.assertLess(abs(r.rounds[-1].z), abs(r.reduced_q) + 2)

    def test_errors_are_decimal(self):
        r = sin_cos(1)
        self.assertIsInstance(r.sin_error, Decimal)
        self.assertIsInstance(r.cos_error, Decimal)

    def test_float_views(self):
        r = sin_cos(PI_Q // 2)
        self.assertAlmostEqual(r.sin, q_to_float(r.sin_q))
        self.assertAlmostEqual(r.cos, q_to_float(r.cos_q))


if __name__ == "__main__":
    unittest.main()

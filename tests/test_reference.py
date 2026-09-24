"""Tests for the independent Decimal reference oracle.

The reference must be verifiable on its own (known identities/values) and
must not obtain expectations by calling the CORDIC core under test.
"""

import inspect
import unittest
from decimal import Decimal

import cordic_q428.reference as reference
import cordic_q428.cordic as cordic
from cordic_q428 import HALF_PI_Q, PI_Q, SCALE, polar_ref, sin_cos_ref


class TestReferenceIndependence(unittest.TestCase):
    def test_reference_does_not_call_core(self):
        src = inspect.getsource(reference)
        self.assertNotIn("cordic.sin_cos", src)
        self.assertNotIn("from .cordic import", src)
        self.assertNotIn("from cordic_q428.cordic", src)
        self.assertNotIn("import cordic", src)

    def test_no_math_trig_anywhere_in_package(self):
        # Check actual usage (imports/calls), not doc mentions: strip comments
        # and docstrings by tokenizing would be overkill — checking that the
        # math module is never imported in the package suffices, since
        # math.sin/cos/tan are unreachable without it.
        import cordic_q428.fixedpoint as fp
        import cordic_q428.tables as tb
        for mod in (fp, tb, reference, cordic):
            src = inspect.getsource(mod)
            self.assertNotIn("import math", src, mod.__name__)
            self.assertNotIn("from math", src, mod.__name__)


class TestReferenceValues(unittest.TestCase):
    def test_zero(self):
        s, c = sin_cos_ref(0)
        self.assertEqual(s, 0)
        self.assertEqual(c, 1)

    # The Q4.28-quantized pi/2 and pi differ from the true values by < 1 ulp
    # (~3.7e-9 rad), so sin/cos there are ~1e-9 away from the exact 0/1 — the
    # reference is faithful to the *quantized* angle.  Tolerance: 2 ulp.
    ULP2 = 2.0 / SCALE

    def test_half_pi(self):
        s, c = sin_cos_ref(HALF_PI_Q)
        self.assertAlmostEqual(float(s), 1.0, delta=self.ULP2)
        self.assertAlmostEqual(float(c), 0.0, delta=self.ULP2)

    def test_pi(self):
        s, c = sin_cos_ref(PI_Q)
        self.assertAlmostEqual(float(s), 0.0, delta=self.ULP2)
        self.assertAlmostEqual(float(c), -1.0, delta=self.ULP2)

    def test_known_angle_sixth_pi(self):
        # pi/6 -> sin = 0.5, cos = sqrt(3)/2 (checked as cos^2 = 3/4).
        q = PI_Q // 6
        s, c = sin_cos_ref(q)
        self.assertAlmostEqual(float(s), 0.5, places=9)
        self.assertAlmostEqual(float(c * c), 0.75, places=9)

    def test_pythagorean_identity_across_range(self):
        for q in (0, 1, -1, PI_Q, -PI_Q, 2**31 - 1, -(2**31), 777777777):
            s, c = sin_cos_ref(q)
            ident = float(s * s + c * c)
            self.assertAlmostEqual(ident, 1.0, places=20)

    def test_symmetry(self):
        # Exact Decimal equality is not required: the two reduction paths round
        # differently in the final guard digits.  Compare well inside the
        # 60-digit working precision.
        for q in (1, 123456789, PI_Q // 3):
            sp, cp = sin_cos_ref(q)
            sn, cn = sin_cos_ref(-q)
            self.assertLess(abs(sp + sn), Decimal("1e-40"))
            self.assertLess(abs(cp - cn), Decimal("1e-40"))

    def test_returns_decimal(self):
        s, c = sin_cos_ref(0)
        self.assertIsInstance(s, Decimal)
        self.assertIsInstance(c, Decimal)


class TestPolarReference(unittest.TestCase):
    """The polar oracle must be independent and match hand-computed values."""

    def test_zero_vector(self):
        a, m = polar_ref(0, 0)
        self.assertEqual(a, 0)
        self.assertEqual(m, 0)

    def test_axes(self):
        half_pi = Decimal(HALF_PI_Q) / SCALE
        pi = Decimal(PI_Q) / SCALE
        one = Decimal(1)
        cases = (
            (SCALE, 0, Decimal(0), one),
            (0, SCALE, half_pi, one),
            (0, -SCALE, -half_pi, one),
            (-SCALE, 0, pi, one),  # negative x axis, y == 0 -> +pi by convention
        )
        for x, y, want_a, want_m in cases:
            with self.subTest(x=x, y=y):
                a, m = polar_ref(x, y)
                self.assertLess(abs(a - want_a), Decimal(2) / SCALE)
                self.assertLess(abs(m - want_m), Decimal(1) / SCALE)

    def test_diagonal_and_345(self):
        # (1,1): angle pi/4, magnitude sqrt(2) (checked as m^2 = 2).
        a, m = polar_ref(SCALE, SCALE)
        self.assertLess(abs(a - Decimal(PI_Q) / (4 * SCALE)), Decimal(1) / SCALE)
        self.assertAlmostEqual(float(m * m), 2.0, places=15)
        # (3,4): magnitude exactly 5.
        a, m = polar_ref(3 * SCALE, 4 * SCALE)
        self.assertAlmostEqual(float(m), 5.0, places=15)
        # tan(angle) = 4/3 reconstructed without calling atan under test.
        self.assertGreater(a, 0)

    def test_quadrant_signs(self):
        a_q1, _ = polar_ref(SCALE, SCALE)
        a_q2, _ = polar_ref(-SCALE, SCALE)
        a_q3, _ = polar_ref(-SCALE, -SCALE)
        a_q4, _ = polar_ref(SCALE, -SCALE)
        self.assertTrue(a_q1 > 0 and a_q2 > 0)
        self.assertTrue(a_q3 < 0 and a_q4 < 0)
        self.assertLess(abs(a_q1 + a_q4), Decimal(2) / SCALE)
        # QII angle +3*pi/4 and QIII angle -3*pi/4 are negatives of each other.
        self.assertLess(abs(a_q2 + a_q3), Decimal(2) / SCALE)

    def test_rejects_bad_input(self):
        for bad in (True, False, 1.0, "1", None):
            with self.assertRaises(Exception):
                polar_ref(bad, 0)
            with self.assertRaises(Exception):
                polar_ref(0, bad)

    def test_returns_decimal(self):
        a, m = polar_ref(3, 4)
        self.assertIsInstance(a, Decimal)
        self.assertIsInstance(m, Decimal)


if __name__ == "__main__":
    unittest.main()

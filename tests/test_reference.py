"""Tests for the independent Decimal reference oracle.

The reference must be verifiable on its own (known identities/values) and
must not obtain expectations by calling the CORDIC core under test.
"""

import inspect
import unittest
from decimal import Decimal

import cordic_q428.reference as reference
import cordic_q428.cordic as cordic
from cordic_q428 import HALF_PI_Q, PI_Q, SCALE, sin_cos_ref


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
        import cordic_q428.envelope as env
        import cordic_q428.fixedpoint as fp
        import cordic_q428.tables as tb
        for mod in (fp, tb, reference, cordic, env):
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


if __name__ == "__main__":
    unittest.main()

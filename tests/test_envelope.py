"""Tests for the conservative error envelope (cordic_q428.envelope).

The envelope is verified three independent ways:

* containment of the fixed-point outputs of ``sin_cos`` (bit-identical
  trajectory, envelope switch on/off);
* containment of the independent Decimal Taylor reference at zero, on both
  sides of every quadrant boundary, and on a fixed set of angles;
* per-round verification of the shift-truncation bounds.
"""

import unittest
from decimal import Decimal
from fractions import Fraction

from cordic_q428 import (
    ATAN_TABLE_Q,
    HALF_PI_Q,
    INT32_MAX,
    INT32_MIN,
    ITERATIONS,
    PI_Q,
    SCALE,
    CordicError,
    EnvelopeResult,
    FixedPointOverflowError,
    InvalidArgumentTypeError,
    compute_envelope,
    sin_cos,
    sin_cos_ref,
)

# Envelope must be far tighter than the whole Q4.28 value range (2**31 ulp)
# for every input — a degenerate full-range "envelope" is a test failure.
MAX_WIDTH_ULP = 4096

# Fixed angle set: zero, +/-1 ulp, both sides of every quadrant boundary,
# a few interior angles, and the signed 32-bit endpoints.
_BOUNDARY_ANGLES = []
for _edge in (0, HALF_PI_Q, PI_Q, HALF_PI_Q + PI_Q, -HALF_PI_Q, -PI_Q, -(HALF_PI_Q + PI_Q)):
    for _delta in (-1, 0, 1):
        _q = _edge + _delta
        if INT32_MIN <= _q <= INT32_MAX:
            _BOUNDARY_ANGLES.append(_q)
FIXED_ANGLES = tuple(
    dict.fromkeys(
        _BOUNDARY_ANGLES
        + [1, -1, 2, -2, 123456789, -123456789, PI_Q // 6, PI_Q // 3,
           268435456, -268435456, INT32_MIN, INT32_MAX, INT32_MIN + 1]
    )
)


def _ref_scaled(angle_q: int) -> tuple[Fraction, Fraction]:
    """Independent Decimal reference, scaled to ulp, as exact rationals."""
    s, c = sin_cos_ref(angle_q)
    return Fraction(s) * SCALE, Fraction(c) * SCALE


class TestEnvelopeContainment(unittest.TestCase):
    """Decimal-oracle containment at zero, quadrant boundaries, fixed angles."""

    def test_contains_fixed_point_outputs(self):
        for q in FIXED_ANGLES:
            with self.subTest(q=q):
                r = sin_cos(q, with_envelope=True)
                e = r.envelope
                self.assertIsNotNone(e)
                self.assertLessEqual(e.sin_lo, r.sin_q)
                self.assertGreaterEqual(e.sin_hi, r.sin_q)
                self.assertLessEqual(e.cos_lo, r.cos_q)
                self.assertGreaterEqual(e.cos_hi, r.cos_q)

    def test_contains_decimal_reference(self):
        for q in FIXED_ANGLES:
            with self.subTest(q=q):
                e = compute_envelope(q)
                rs, rc = _ref_scaled(q)
                self.assertLessEqual(e.sin_lo_exact, rs)
                self.assertGreaterEqual(e.sin_hi_exact, rs)
                self.assertLessEqual(e.cos_lo_exact, rc)
                self.assertGreaterEqual(e.cos_hi_exact, rc)
                # Same check through the public convenience predicate.
                self.assertTrue(e.contains(rs, rc))

    def test_integer_bounds_bracket_exact_bounds(self):
        e = compute_envelope(PI_Q // 3)
        self.assertLessEqual(e.sin_lo, e.sin_lo_exact)
        self.assertGreaterEqual(e.sin_hi, e.sin_hi_exact)
        self.assertLessEqual(e.cos_lo, e.cos_lo_exact)
        self.assertGreaterEqual(e.cos_hi, e.cos_hi_exact)
        for v in (e.sin_lo, e.sin_hi, e.cos_lo, e.cos_hi):
            self.assertIsInstance(v, int)

    def test_envelope_is_not_degenerate(self):
        # Conservative is fine; the whole value range is not.
        for q in FIXED_ANGLES:
            with self.subTest(q=q):
                e = compute_envelope(q)
                self.assertGreater(e.sin_width_ulp, 0)
                self.assertGreater(e.cos_width_ulp, 0)
                self.assertLess(e.sin_width_ulp, MAX_WIDTH_ULP)
                self.assertLess(e.cos_width_ulp, MAX_WIDTH_ULP)

    def test_full_range_sweep_containment(self):
        # Deterministic 101-point sweep across the whole representable range:
        # every sample must contain both the core outputs and the reference.
        for k in range(101):
            q = INT32_MIN + k * (2**32 - 1) // 100
            with self.subTest(q=q):
                r = sin_cos(q, with_envelope=True)
                e = r.envelope
                rs, rc = _ref_scaled(q)
                self.assertTrue(e.contains(r.sin_q, r.cos_q), f"fp q={q}")
                self.assertTrue(e.contains(rs, rc), f"ref q={q}")
                self.assertLess(e.sin_width_ulp, MAX_WIDTH_ULP)
                self.assertLess(e.cos_width_ulp, MAX_WIDTH_ULP)


class TestPerRoundTruncationBound(unittest.TestCase):
    """Per-round verification of the arithmetic-shift truncation bound."""

    def test_shift_truncation_bound_each_round(self):
        for q in (0, 1, -1, HALF_PI_Q, PI_Q, INT32_MIN, INT32_MAX, 999999937):
            with self.subTest(q=q):
                e = compute_envelope(q)
                self.assertEqual(len(e.rounds), ITERATIONS)
                for i, rd in enumerate(e.rounds):
                    self.assertEqual(rd.index, i)
                    # sar truncation loses v/2**i - floor(v/2**i) < 1 - 2**-i.
                    bound = Fraction(1) - Fraction(1, 1 << rd.index)
                    self.assertGreaterEqual(rd.shift_trunc, 0)
                    self.assertLessEqual(rd.shift_trunc, bound)

    def test_round_widths_nonnegative_and_finite(self):
        e = compute_envelope(123456789)
        for rd in e.rounds:
            self.assertGreaterEqual(rd.x_width, 0)
            self.assertGreaterEqual(rd.y_width, 0)
            self.assertGreaterEqual(rd.z_width, 0)
            self.assertLess(rd.x_width, MAX_WIDTH_ULP * 4)
            self.assertLess(rd.y_width, MAX_WIDTH_ULP * 4)

    def test_direction_branch_union_occurs(self):
        # Some angle must exercise the z-straddles-zero union path; whenever
        # it happens the directions record must show both branches.
        saw_split = False
        for q in list(range(0, 33)) + [PI_Q, -PI_Q, HALF_PI_Q]:
            e = compute_envelope(q)
            for rd in e.rounds:
                self.assertIn(rd.directions, ((1,), (-1,), (1, -1)))
                if len(rd.directions) == 2:
                    saw_split = True
        self.assertTrue(saw_split, "no branch-boundary union was exercised")


class TestErrorSources(unittest.TestCase):
    def test_sources_nonnegative_and_within_total(self):
        for q in (0, 1, HALF_PI_Q, PI_Q, INT32_MIN, 123456789):
            with self.subTest(q=q):
                e = compute_envelope(q)
                total_sin = e.sin_width_ulp / 2
                total_cos = e.cos_width_ulp / 2
                for src in (e.sources.gain_quantization,
                            e.sources.table_quantization,
                            e.sources.shift_truncation):
                    self.assertGreaterEqual(src.sin, 0)
                    self.assertGreaterEqual(src.cos, 0)
                    self.assertLessEqual(src.sin, total_sin)
                    self.assertLessEqual(src.cos, total_cos)

    def test_gain_source_is_small(self):
        # Gain quantization starts at <= 0.5 ulp and is only amplified by the
        # CORDIC gain (~1.647): it can never dominate the envelope.
        e = compute_envelope(PI_Q // 3)
        self.assertLess(e.sources.gain_quantization.sin, 4)
        self.assertLess(e.sources.gain_quantization.cos, 4)

    def test_table_and_shift_sources_present(self):
        e = compute_envelope(PI_Q // 3)
        self.assertGreater(e.sources.table_quantization.sin, 0)
        self.assertGreater(e.sources.shift_truncation.sin, 0)


class TestSwitchKeepsCoreBitIdentical(unittest.TestCase):
    def test_envelope_off_by_default(self):
        r = sin_cos(PI_Q // 3)
        self.assertIsNone(r.envelope)

    def test_results_identical_with_switch(self):
        for q in FIXED_ANGLES:
            with self.subTest(q=q):
                a = sin_cos(q)
                b = sin_cos(q, with_envelope=True)
                self.assertEqual(a.sin_q, b.sin_q)
                self.assertEqual(a.cos_q, b.cos_q)
                self.assertEqual(a.reduced_q, b.reduced_q)
                self.assertEqual(a.sign_flip, b.sign_flip)
                self.assertEqual(a.rounds, b.rounds)
                self.assertEqual(a.sin_error, b.sin_error)
                self.assertEqual(a.cos_error, b.cos_error)

    def test_envelope_metadata_matches_core(self):
        for q in FIXED_ANGLES:
            with self.subTest(q=q):
                r = sin_cos(q, with_envelope=True)
                e = r.envelope
                self.assertEqual(e.angle_q, q)
                self.assertEqual(e.reduced_q, r.reduced_q)
                self.assertEqual(e.sign_flip, r.sign_flip)

    def test_envelope_result_structure(self):
        e = compute_envelope(1)
        self.assertIsInstance(e, EnvelopeResult)
        self.assertIsInstance(e.sin_lo_exact, Fraction)
        self.assertEqual(len(e.rounds), ITERATIONS)
        self.assertIsInstance(e.sin_width_ulp, Fraction)


class TestEnvelopeErrorSemantics(unittest.TestCase):
    def test_bool_and_non_int_rejected(self):
        for bad in (True, False, 1.5, "1", None):
            with self.assertRaises(InvalidArgumentTypeError, msg=f"{bad!r}"):
                compute_envelope(bad)
            with self.assertRaises(InvalidArgumentTypeError, msg=f"{bad!r}"):
                sin_cos(bad, with_envelope=True)

    def test_out_of_range_rejected_not_saturated(self):
        for bad in (INT32_MAX + 1, INT32_MIN - 1):
            with self.assertRaises(FixedPointOverflowError):
                compute_envelope(bad)
            with self.assertRaises(FixedPointOverflowError):
                sin_cos(bad, with_envelope=True)

    def test_errors_share_base(self):
        try:
            compute_envelope(True)
        except CordicError:
            pass
        else:
            self.fail("expected CordicError")

    def test_no_partial_state_after_failure(self):
        good = sin_cos(PI_Q // 3, with_envelope=True)
        with self.assertRaises(CordicError):
            sin_cos(INT32_MAX + 1, with_envelope=True)
        self.assertEqual(good, sin_cos(PI_Q // 3, with_envelope=True))


if __name__ == "__main__":
    unittest.main()

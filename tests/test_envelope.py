"""Tests for the conservative error envelope (cordic_q428.envelope).

Containment is checked independently against the Decimal Taylor oracle at
zero, on both sides of every quadrant boundary, and on a fixed set of
angles; the per-round truncation bounds are verified round by round against
the actual register trace.
"""

import unittest
from decimal import Decimal
from fractions import Fraction

from cordic_q428 import (
    GAIN_INV_Q,
    HALF_PI_Q,
    INT32_MAX,
    INT32_MIN,
    ITERATIONS,
    PI_Q,
    SCALE,
    CordicError,
    SinCosEnvelope,
    float_to_q,
    sin_cos,
    sin_cos_ref,
    sin_cos_with_envelope,
)
from cordic_q428.envelope import GAIN_ERR_ULP, TABLE_ERR_ULP

# Fixed angle set: zero, min step, quadrant edges +/- 1 ulp, mid-quadrant
# angles, and the full-range endpoints.
FIXED_ANGLES = (
    0,
    1,
    -1,
    HALF_PI_Q - 1,
    HALF_PI_Q,
    HALF_PI_Q + 1,
    -HALF_PI_Q - 1,
    -HALF_PI_Q,
    -HALF_PI_Q + 1,
    PI_Q - 1,
    PI_Q,
    -PI_Q,
    -PI_Q + 1,
    HALF_PI_Q + PI_Q,
    PI_Q // 6,
    PI_Q // 3,
    -PI_Q // 3,
    float_to_q(1.0),
    123456789,
    -987654321,
    INT32_MIN,
    INT32_MAX,
)


def _ref_contains(env: SinCosEnvelope, angle_q: int) -> bool:
    """Independent check: Decimal oracle value lies inside the envelope."""
    ref_sin, ref_cos = sin_cos_ref(angle_q)
    return (
        env.sin_lo <= Fraction(ref_sin) * SCALE <= env.sin_hi
        and env.cos_lo <= Fraction(ref_cos) * SCALE <= env.cos_hi
    )


class TestEnvelopeContainment(unittest.TestCase):
    """Envelope contains both the integer output and the Decimal oracle."""

    def test_fixed_angles(self):
        for q in FIXED_ANGLES:
            with self.subTest(q=q):
                r = sin_cos(q, with_envelope=True)
                env = r.envelope
                self.assertIsNotNone(env)
                self.assertTrue(
                    env.contains(r.sin_q, r.cos_q),
                    f"output outside envelope: q={q} "
                    f"sin [{env.sin_lo},{env.sin_hi}] got {r.sin_q}; "
                    f"cos [{env.cos_lo},{env.cos_hi}] got {r.cos_q}",
                )
                self.assertTrue(_ref_contains(env, q), f"reference outside envelope: q={q}")

    def test_sweep_containment(self):
        # Deterministic grid across the full representable range.
        for k in range(101):
            q = INT32_MIN + k * (2**32 - 1) // 100
            with self.subTest(q=q):
                r = sin_cos(q, with_envelope=True)
                self.assertTrue(r.envelope.contains(r.sin_q, r.cos_q))
                self.assertTrue(_ref_contains(r.envelope, q))

    def test_envelope_not_full_range(self):
        # The envelope must be a real proof, not the whole domain: for a
        # generic angle it is far tighter than the 2**32-wide value range,
        # yet wide enough to be a sound conservative bound.
        r = sin_cos(float_to_q(1.0), with_envelope=True)
        env = r.envelope
        self.assertGreater(env.sin_width, 0)
        self.assertLess(env.sin_width, 2**20)
        self.assertLess(env.cos_width, 2**20)
        # And it must bound the actual observed error.
        self.assertGreaterEqual(env.sin_width, abs(r.sin_error) * SCALE)
        self.assertGreaterEqual(env.cos_width, abs(r.cos_error) * SCALE)


class TestSwitchTransparency(unittest.TestCase):
    """The envelope switch adds proof information only."""

    def test_switch_off_leaves_envelope_none(self):
        self.assertIsNone(sin_cos(PI_Q // 3).envelope)

    def test_outputs_bit_identical(self):
        for q in FIXED_ANGLES:
            with self.subTest(q=q):
                plain = sin_cos(q)
                armed = sin_cos(q, with_envelope=True)
                self.assertEqual(plain.sin_q, armed.sin_q)
                self.assertEqual(plain.cos_q, armed.cos_q)
                self.assertEqual(plain.reduced_q, armed.reduced_q)
                self.assertEqual(plain.sign_flip, armed.sign_flip)
                self.assertEqual(plain.rounds, armed.rounds)
                self.assertEqual(plain.sin_error, armed.sin_error)
                self.assertEqual(plain.cos_error, armed.cos_error)

    def test_named_convenience_matches_switch(self):
        a = sin_cos(PI_Q // 5, with_envelope=True)
        b = sin_cos_with_envelope(PI_Q // 5)
        self.assertEqual(a, b)
        self.assertIsNotNone(b.envelope)

    def test_rejections_still_raised_with_switch(self):
        with self.assertRaises(CordicError):
            sin_cos(True, with_envelope=True)
        with self.assertRaises(CordicError):
            sin_cos(INT32_MAX + 1, with_envelope=True)


class TestPerRoundTruncationBounds(unittest.TestCase):
    """Round-by-round verification of the shift-truncation bound and of
    register containment inside the propagated intervals."""

    def test_truncation_bound_each_round(self):
        for q in (0, 1, -1, float_to_q(1.0), -HALF_PI_Q, INT32_MIN, 123456789):
            with self.subTest(q=q):
                r = sin_cos(q, with_envelope=True)
                x_old, y_old = GAIN_INV_Q, 0
                for i, st in enumerate(r.rounds):
                    for v in (x_old, y_old):
                        # sar truncation f = v/2**i - floor(v/2**i) must obey
                        # 0 <= f <= 1 - 2**-i exactly as the engine assumes.
                        f = Fraction(v, 1 << i) - Fraction(v >> i)
                        self.assertGreaterEqual(f, 0)
                        self.assertLessEqual(f, Fraction((1 << i) - 1, 1 << i))
                    x_old, y_old = st.x, st.y

    def test_registers_inside_intervals_each_round(self):
        for q in (0, 1, float_to_q(1.0), HALF_PI_Q, -PI_Q, INT32_MAX):
            with self.subTest(q=q):
                r = sin_cos(q, with_envelope=True)
                env = r.envelope
                self.assertEqual(len(env.rounds), ITERATIONS)
                for i, (st, re) in enumerate(zip(r.rounds, env.rounds)):
                    self.assertEqual(re.index, i)
                    # The register's actual direction must be one of the
                    # directions the engine propagated.
                    self.assertIn(st.direction, re.directions)
                    self.assertIn(st.x, re.x, f"round {i} x")
                    self.assertIn(st.y, re.y, f"round {i} y")
                    self.assertIn(st.z, re.z, f"round {i} z")

    def test_union_rounds_are_straddles(self):
        # Wherever the engine unioned, the pre-round z interval must have
        # straddled 0 (both directions genuinely possible).
        r = sin_cos(float_to_q(1.0), with_envelope=True)
        env = r.envelope
        prev = None
        for re in env.rounds:
            if re.index in env.union_rounds:
                self.assertEqual(re.directions, (1, -1))
                self.assertLessEqual(prev.z.lo, 0)
                self.assertGreaterEqual(prev.z.hi, 0)
            else:
                self.assertEqual(len(re.directions), 1)
            prev = re
        self.assertGreater(len(env.union_rounds), 0)  # not a vacuous test


class TestErrorSourceAccounting(unittest.TestCase):
    def test_quantization_errors_are_exact_and_small(self):
        self.assertEqual(len(TABLE_ERR_ULP, ), ITERATIONS)
        for e in TABLE_ERR_ULP:
            self.assertGreater(e, 0)
            self.assertLessEqual(e, Fraction(1, 2) + Fraction(1, 10**30))
        self.assertGreater(GAIN_ERR_ULP, 0)
        self.assertLessEqual(GAIN_ERR_ULP, Fraction(1, 2) + Fraction(1, 10**30))

    def test_sources_nonnegative_and_within_total(self):
        env = sin_cos(float_to_q(1.0), with_envelope=True).envelope
        src = env.sources
        total = max(env.sin_width, env.cos_width)
        for name in (
            "gain_quantization_ulp",
            "table_quantization_ulp",
            "shift_truncation_ulp",
            "residual_rotation_ulp",
        ):
            v = getattr(src, name)
            self.assertIsInstance(v, Fraction)
            self.assertGreaterEqual(v, 0, name)
            # Monotonicity: one source alone cannot exceed all sources.
            self.assertLessEqual(v, total, name)
        # Shift truncation and table quantization must actually contribute
        # (a vacuous zero accounting would be a bug).
        self.assertGreater(src.shift_truncation_ulp, 0)
        self.assertGreater(src.table_quantization_ulp, 0)

    def test_gain_source_isolated_magnitude(self):
        # Gain quantization alone cannot move the endpoint by more than the
        # gain error itself (rotations preserve magnitude), plus slack for
        # the residual rotation of the widened z interval.
        env = sin_cos(float_to_q(1.0), with_envelope=True).envelope
        self.assertLessEqual(
            env.sources.gain_quantization_ulp,
            GAIN_ERR_ULP + env.sources.residual_rotation_ulp + 1,
        )


class TestQuadrantBoundaryUnion(unittest.TestCase):
    def test_boundary_angles_use_both_paths(self):
        for q in (HALF_PI_Q, -HALF_PI_Q):
            with self.subTest(q=q):
                env = sin_cos(q, with_envelope=True).envelope
                self.assertTrue(env.boundary_union)
                r = sin_cos(q)
                self.assertTrue(env.contains(r.sin_q, r.cos_q))
                self.assertTrue(_ref_contains(env, q))

    def test_off_boundary_angles_use_single_path(self):
        for q in (0, HALF_PI_Q - 1, PI_Q, INT32_MIN):
            with self.subTest(q=q):
                env = sin_cos(q, with_envelope=True).envelope
                self.assertFalse(env.boundary_union)

    def test_one_ulp_past_boundary_reduces_onto_boundary(self):
        # HALF_PI_Q + 1 reduces to exactly -HALF_PI_Q: the union must engage
        # on the *reduced* angle, and containment must still hold.
        r = sin_cos(HALF_PI_Q + 1, with_envelope=True)
        self.assertEqual(r.reduced_q, -HALF_PI_Q)
        self.assertTrue(r.envelope.boundary_union)
        self.assertTrue(r.envelope.contains(r.sin_q, r.cos_q))
        self.assertTrue(_ref_contains(r.envelope, HALF_PI_Q + 1))


class TestEnvelopeStructure(unittest.TestCase):
    def test_bounds_are_ints_and_ordered(self):
        env = sin_cos(PI_Q // 3, with_envelope=True).envelope
        for v in (env.sin_lo, env.sin_hi, env.cos_lo, env.cos_hi):
            self.assertIsInstance(v, int)
        self.assertLessEqual(env.sin_lo, env.sin_hi)
        self.assertLessEqual(env.cos_lo, env.cos_hi)

    def test_envelope_is_deterministic(self):
        a = sin_cos(PI_Q // 3, with_envelope=True).envelope
        b = sin_cos(PI_Q // 3, with_envelope=True).envelope
        self.assertEqual(a, b)

    def test_result_equality_unaffected_by_default_field(self):
        # Two plain results still compare equal; the envelope field does not
        # disturb the existing equality contract.
        self.assertEqual(sin_cos(0), sin_cos(0))
        self.assertNotEqual(sin_cos(0), sin_cos(0, with_envelope=True))


if __name__ == "__main__":
    unittest.main()

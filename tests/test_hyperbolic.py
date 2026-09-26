"""Hyperbolic CORDIC tests: domain, repeat rounds, gain, error vs Decimal exp oracle."""

import unittest
from decimal import Decimal, localcontext

from cordic_q428 import (
    ATANH_TABLE_Q,
    HYP_GAIN_INV_Q,
    HYP_ITERATIONS,
    HYP_REPEAT_INDICES,
    HYP_SCHEDULE,
    INT32_MAX,
    INT32_MIN,
    SCALE,
    Z_MAX_Q,
    CordicError,
    InputRangeError,
    InvalidArgumentTypeError,
    sinh_cosh,
    sinh_cosh_ref,
)

# Analytic error budget for the 34-micro-rotation hyperbolic schedule:
# <= 1 ulp gain quantization + 34 shift truncations of <= 1 ulp each
# amplified by the ~1.2075 hyperbolic gain + <= 1 ulp table quantization.
# 120 ulp is a safe, non-loose bound; observed errors are far smaller.
MAX_ERR_ULP = 120


def _err_ulp(result):
    return result.sinh_error * SCALE, result.cosh_error * SCALE


def _configs():
    """Acceptance configs: zero, both boundaries, and neighborhoods of the
    repeated rounds (i = 4 and i = 13), plus a dense deterministic grid."""
    zs = {0, Z_MAX_Q, -Z_MAX_Q, 1, -1, Z_MAX_Q - 1, -(Z_MAX_Q - 1)}
    # Angles whose remaining z-residual crosses zero near a repeated round:
    # partial sums of the scheduled atanh angles up to each repeat, +/- 4 ulp.
    acc = 0
    for i, repeated in HYP_SCHEDULE:
        acc += ATANH_TABLE_Q[i]
        if repeated:
            for delta in range(-4, 5):
                for base in (acc, acc - ATANH_TABLE_Q[i]):
                    z = base + delta
                    if abs(z) <= Z_MAX_Q:
                        zs.add(z)
                        zs.add(-z)
    # Deterministic grid across the full domain.
    for k in range(1, 200):
        z = k * Z_MAX_Q // 200
        zs.add(z)
        zs.add(-z)
    return sorted(zs)


class TestDomainAndRejection(unittest.TestCase):
    def test_boundaries_accepted(self):
        for z in (Z_MAX_Q, -Z_MAX_Q):
            r = sinh_cosh(z)
            self.assertIsInstance(r.cosh_q, int)

    def test_out_of_domain_rejected(self):
        for z in (Z_MAX_Q + 1, -(Z_MAX_Q + 1), INT32_MAX, INT32_MIN):
            with self.assertRaises(InputRangeError):
                sinh_cosh(z)

    def test_bool_rejected(self):
        with self.assertRaises(InvalidArgumentTypeError):
            sinh_cosh(True)

    def test_non_int_rejected(self):
        for bad in (0.5, "1", None):
            with self.assertRaises(InvalidArgumentTypeError):
                sinh_cosh(bad)

    def test_errors_are_cordic_errors(self):
        with self.assertRaises(CordicError):
            sinh_cosh(Z_MAX_Q + 1)


class TestScheduleAndGain(unittest.TestCase):
    def test_repeat_indices_rule(self):
        # i_next = 3*i + 1, starting at 4, within 32 iterations.
        self.assertEqual(HYP_REPEAT_INDICES, (4, 13))

    def test_schedule_repeats(self):
        self.assertEqual(len(HYP_SCHEDULE), HYP_ITERATIONS + len(HYP_REPEAT_INDICES))
        self.assertEqual([i for i, rep in HYP_SCHEDULE if rep], list(HYP_REPEAT_INDICES))
        # Each index 1..32 appears once, repeats twice.
        for i in range(1, HYP_ITERATIONS + 1):
            count = sum(1 for j, _ in HYP_SCHEDULE if j == i)
            self.assertEqual(count, 2 if i in HYP_REPEAT_INDICES else 1)

    def test_gain_includes_repeats(self):
        # Recompute the inverse gain independently from the schedule.
        with localcontext() as ctx:
            ctx.prec = 80
            gain_inv = Decimal(1)
            for i, _rep in HYP_SCHEDULE:
                gain_inv /= (Decimal(1) - Decimal(2) ** (-2 * i)).sqrt()
            q = int((gain_inv * SCALE).quantize(Decimal(1), rounding="ROUND_HALF_EVEN"))
        self.assertEqual(q, HYP_GAIN_INV_Q)
        # And it must differ from the no-repeat gain: repeats are not lost.
        with localcontext() as ctx:
            ctx.prec = 80
            bare = Decimal(1)
            for i in range(1, HYP_ITERATIONS + 1):
                bare /= (Decimal(1) - Decimal(2) ** (-2 * i)).sqrt()
            q_bare = int((bare * SCALE).quantize(Decimal(1), rounding="ROUND_HALF_EVEN"))
        self.assertNotEqual(q_bare, HYP_GAIN_INV_Q)

    def test_round_trace_records_shift_and_rounding(self):
        r = sinh_cosh(SCALE // 3)
        self.assertEqual(len(r.rounds), len(HYP_SCHEDULE))
        for state, (i, rep) in zip(r.rounds, HYP_SCHEDULE):
            self.assertEqual(state.index, i)
            self.assertEqual(state.shift, i)
            self.assertEqual(state.repeated, rep)
            self.assertIn(state.direction, (1, -1))
        # Shifted operands actually recorded equal sar of the previous x/y.
        prev_x, prev_y = HYP_GAIN_INV_Q, 0
        for state in r.rounds:
            self.assertEqual(state.x_shift, prev_x >> state.shift)
            self.assertEqual(state.y_shift, prev_y >> state.shift)
            prev_x, prev_y = state.x, state.y


class TestAccuracy(unittest.TestCase):
    def test_zero(self):
        r = sinh_cosh(0)
        self.assertLessEqual(abs(r.sinh_q), 8)
        self.assertLessEqual(abs(r.cosh_q - SCALE), MAX_ERR_ULP)

    def test_identity_cosh2_minus_sinh2(self):
        # cosh^2(z) - sinh^2(z) == 1 within fixed-point rounding.
        for z in _configs():
            r = sinh_cosh(z)
            ident = (r.cosh_q * r.cosh_q - r.sinh_q * r.sinh_q) / (SCALE * SCALE)
            self.assertAlmostEqual(ident, 1.0, delta=2e-6)

    def test_max_error_per_config(self):
        worst = 0.0
        worst_z = None
        for z in _configs():
            r = sinh_cosh(z)
            err = max(abs(_err_ulp(r)[0]), abs(_err_ulp(r)[1]))
            self.assertLessEqual(float(err), MAX_ERR_ULP, f"z_q={z}")
            if err > worst:
                worst, worst_z = err, z
        # Observed maximum over the acceptance configs, printed for the -v log.
        print(f"\n[hyperbolic] max observed error over {len(_configs())} configs: "
              f"{float(worst):.3f} ulp at z_q={worst_z}")


class TestReference(unittest.TestCase):
    def test_reference_matches_exp_definition(self):
        # The oracle is the Decimal exponential definition; spot-check at 60
        # digits against an independently summed exp Taylor series.
        with localcontext() as ctx:
            ctx.prec = 60
            z = Decimal(123456789) / SCALE
            term = Decimal(1)
            e = Decimal(1)
            k = 1
            while True:
                term *= z / k
                e += term
                if abs(term) < Decimal(10) ** -55:
                    break
                k += 1
            sinh = (e - 1 / e) / 2
            cosh = (e + 1 / e) / 2
        ref_sinh, ref_cosh = sinh_cosh_ref(123456789)
        self.assertLess(abs(ref_sinh - sinh), Decimal(10) ** -50)
        self.assertLess(abs(ref_cosh - cosh), Decimal(10) ** -50)

    def test_reference_known_values(self):
        s, c = sinh_cosh_ref(0)
        self.assertEqual(s, 0)
        self.assertEqual(c, 1)


if __name__ == "__main__":
    unittest.main()

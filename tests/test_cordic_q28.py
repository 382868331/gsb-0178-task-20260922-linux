"""cordic_q28 的单元测试。

独立参考（cordic_q28.reference）由 Decimal 泰勒级数独立实现，
不调用被测核心得到期望值；往返一致性（sin^2+cos^2≈1）仅作补充检查。
"""

import math
import unittest
from decimal import Decimal, localcontext

import cordic_q28 as cq
from cordic_q28.core import _ashr, _check_int32
from cordic_q28.tables import PI, quantize_q28

# 采样误差容限（单位：Q4.28 ulp）。32 轮量化表 CORDIC 的固有误差为十几 ulp
# 量级（表项量化 ±0.5 ulp 累积 + 地板移位偏置），取 32 ulp 作为带裕量的
# 硬上界；该界不用于掩盖边界行为——边界用例另有精确断言。
TOL_LSB = Decimal(32)


def angles_to_test():
    """确定性样本：边界 + 伪随机（固定种子）满范围角度。"""
    import random

    rng = random.Random(20260922)
    samples = [
        0,
        1,
        -1,
        cq.HALF_PI_Q,
        -cq.HALF_PI_Q,
        cq.PI_Q,
        -cq.PI_Q,
        cq.TWO_PI_Q,
        -cq.TWO_PI_Q,
        cq.HALF_PI_Q + 1,
        cq.HALF_PI_Q - 1,
        cq.INT32_MIN,
        cq.INT32_MAX,
        cq.INT32_MIN + 1,
        cq.INT32_MAX - 1,
    ]
    samples += [rng.randrange(cq.INT32_MIN, cq.INT32_MAX + 1) for _ in range(120)]
    return samples


class TestInputValidation(unittest.TestCase):
    def test_reject_bool(self):
        for bad in (True, False):
            with self.assertRaises(cq.InvalidInputError):
                cq.sin_cos(bad)

    def test_reject_nan_and_inf(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(cq.InvalidInputError):
                cq.sin_cos(bad)

    def test_reject_float_even_integral(self):
        with self.assertRaises(cq.InvalidInputError):
            cq.sin_cos(1.0)

    def test_reject_non_numeric_types(self):
        for bad in ("1", None, [0], object()):
            with self.assertRaises(cq.InvalidInputError):
                cq.sin_cos(bad)

    def test_reject_out_of_range_int(self):
        with self.assertRaises(cq.AngleRangeError):
            cq.sin_cos(cq.INT32_MAX + 1)
        with self.assertRaises(cq.AngleRangeError):
            cq.sin_cos(cq.INT32_MIN - 1)

    def test_error_types_are_locatable(self):
        # 错误同时挂在标准异常层级上，便于调用方分类捕获
        self.assertIsInstance(cq.InvalidInputError("x"), TypeError)
        self.assertIsInstance(cq.AngleRangeError("x"), ValueError)
        self.assertIsInstance(cq.CordicOverflowError("x"), OverflowError)
        for exc in (cq.InvalidInputError, cq.AngleRangeError, cq.CordicOverflowError):
            self.assertTrue(issubclass(exc, cq.CordicError))


class TestTables(unittest.TestCase):
    def test_atan_table_matches_decimal_series(self):
        # 独立重算：atan(2^-i) 量化值必须与表一致（ties-to-even）
        with localcontext() as ctx:
            ctx.prec = 80
            for i, entry in enumerate(cq.ATAN_TABLE):
                x = Decimal(1) / (1 << i)
                # 用足够多項的泰勒级数独立重算（i=0 用 pi/4）
                if i == 0:
                    expected = quantize_q28(PI / 4)
                else:
                    x2 = x * x
                    term = x
                    total = term
                    k = 1
                    while True:
                        term *= x2
                        add = term / (2 * k + 1)
                        total = total - add if k % 2 else total + add
                        if add == 0 or add.adjusted() < -70:
                            break
                        k += 1
                    expected = quantize_q28(+total)
                self.assertEqual(entry, expected, f"ATAN_TABLE[{i}]")

    def test_atan_table_monotone_nonnegative(self):
        self.assertEqual(len(cq.ATAN_TABLE), cq.ITERATIONS)
        for a, b in zip(cq.ATAN_TABLE, cq.ATAN_TABLE[1:]):
            self.assertGreaterEqual(a, b)
            self.assertGreaterEqual(b, 0)
        # i <= 28 时 atan(2**-i) >= 1 ulp；i = 29..31 时不足 1 ulp（含 0.5 ulp
        # 平局取偶），量化为 0
        self.assertTrue(all(v > 0 for v in cq.ATAN_TABLE[:29]))
        self.assertEqual(cq.ATAN_TABLE[29:], (0, 0, 0))

    def test_gain_inverse_value(self):
        # K_INV ≈ 0.6072529350088814；量化后应在 1 ulp 内
        with localcontext() as ctx:
            ctx.prec = 60
            k = Decimal(1)
            for i in range(cq.ITERATIONS):
                k *= (1 + Decimal(1) / (1 << (2 * i))).sqrt()
            ref = quantize_q28(1 / k)
        self.assertEqual(cq.K_INV_Q, ref)
        self.assertAlmostEqual(cq.K_INV_Q / cq.SCALE, 0.6072529350088814, places=8)

    def test_pi_constants_consistent(self):
        self.assertEqual(cq.TWO_PI_Q, quantize_q28(2 * PI))
        self.assertEqual(cq.HALF_PI_Q, quantize_q28(PI / 2))
        # pi 的 Machin 值与已知位数一致
        self.assertTrue(str(cq.PI).startswith("3.141592653589793238462643383279"))


class TestArithmeticShift(unittest.TestCase):
    def test_right_shift_is_floor_for_negatives(self):
        # 算术右移必须向负无穷取整
        self.assertEqual(_ashr(-3, 1), -2)
        self.assertEqual(_ashr(-1, 1), -1)
        self.assertEqual(_ashr(-5, 2), -2)
        self.assertEqual(_ashr(5, 2), 1)
        self.assertEqual(_ashr(-(1 << 28), 28), -1)

    def test_overflow_guard(self):
        _check_int32(cq.INT32_MIN, "t")
        _check_int32(cq.INT32_MAX, "t")
        with self.assertRaises(cq.CordicOverflowError):
            _check_int32(cq.INT32_MAX + 1, "t")
        with self.assertRaises(cq.CordicOverflowError):
            _check_int32(cq.INT32_MIN - 1, "t")


class TestRangeReduction(unittest.TestCase):
    def test_reduced_within_half_pi(self):
        for a in angles_to_test():
            reduced, sign = cq.reduce_angle(a)
            self.assertLessEqual(abs(reduced), cq.HALF_PI_Q + 1)
            self.assertIn(sign, (1, -1))

    def test_quadrant_sign(self):
        # 第一象限角不变号；第二象限（pi/2..pi）取负
        self.assertEqual(cq.reduce_angle(cq.SCALE)[1], 1)  # 1 rad
        self.assertEqual(cq.reduce_angle(cq.SCALE * 2)[1], -1)  # 2 rad
        self.assertEqual(cq.reduce_angle(-cq.SCALE * 2)[1], -1)  # -2 rad
        self.assertEqual(cq.reduce_angle(-cq.SCALE)[1], 1)  # -1 rad

    def test_zero_reduces_to_zero(self):
        self.assertEqual(cq.reduce_angle(0), (0, 1))


class TestCordicCore(unittest.TestCase):
    def test_trace_length_and_final_state(self):
        res = cq.sin_cos(cq.SCALE)
        self.assertEqual(len(res.trace), cq.ITERATIONS)
        last = res.trace[-1]
        # 最终态乘以象限符号必须等于输出
        self.assertEqual(res.sin_q, res.quadrant_sign * last.y)
        self.assertEqual(res.cos_q, res.quadrant_sign * last.x)
        for i, st in enumerate(res.trace):
            self.assertEqual(st.iteration, i)
            self.assertIn(st.direction, (1, -1))
            self.assertTrue(cq.INT32_MIN <= st.x <= cq.INT32_MAX)
            self.assertTrue(cq.INT32_MIN <= st.y <= cq.INT32_MAX)
            self.assertTrue(cq.INT32_MIN <= st.z <= cq.INT32_MAX)

    def test_zero_angle(self):
        res = cq.sin_cos(0)
        self.assertLessEqual(abs(res.sin_q), 8)  # 接近 0
        self.assertLessEqual(abs(res.cos_q - cq.SCALE), 8)  # 接近 1

    def test_min_angle_step(self):
        # 最小角步 2**-28 rad：sin ≈ ±1 ulp，cos ≈ 1
        pos = cq.sin_cos(1)
        neg = cq.sin_cos(-1)
        self.assertLessEqual(abs(pos.sin_q - 1), 2)
        self.assertLessEqual(abs(neg.sin_q + 1), 2)
        self.assertLessEqual(abs(pos.cos_q - cq.SCALE), 8)
        self.assertLessEqual(abs(neg.cos_q - cq.SCALE), 8)

    def test_quadrant_boundaries(self):
        # 象限边界：±pi/2、±pi、±2pi 的量化角，与独立参考对比
        for a in (
            cq.HALF_PI_Q,
            -cq.HALF_PI_Q,
            cq.PI_Q,
            -cq.PI_Q,
            cq.TWO_PI_Q,
            -cq.TWO_PI_Q,
            cq.HALF_PI_Q + cq.PI_Q,  # 3pi/2 附近
            cq.HALF_PI_Q - 1,
            cq.HALF_PI_Q + 1,
        ):
            ev = cq.evaluate(a)
            self.assertLessEqual(abs(ev.sin_error_lsb), TOL_LSB, f"sin @ {a}")
            self.assertLessEqual(abs(ev.cos_error_lsb), TOL_LSB, f"cos @ {a}")

    def test_full_range_inputs(self):
        for a in (cq.INT32_MIN, cq.INT32_MAX, cq.INT32_MIN + 1, cq.INT32_MAX - 1):
            ev = cq.evaluate(a)
            self.assertLessEqual(abs(ev.sin_error_lsb), TOL_LSB, f"sin @ {a}")
            self.assertLessEqual(abs(ev.cos_error_lsb), TOL_LSB, f"cos @ {a}")

    def test_sampled_error_bound(self):
        worst = Decimal(0)
        for a in angles_to_test():
            ev = cq.evaluate(a)
            worst = max(worst, abs(ev.sin_error_lsb), abs(ev.cos_error_lsb))
        self.assertLessEqual(worst, TOL_LSB)

    def test_pythagorean_consistency(self):
        # 补充性往返一致检查（非期望值来源）：sin^2+cos^2 ≈ 1（Q4.28 域）
        for a in angles_to_test()[:40]:
            res = cq.sin_cos(a)
            with localcontext() as ctx:
                ctx.prec = 50
                s = Decimal(res.sin_q) / cq.SCALE
                c = Decimal(res.cos_q) / cq.SCALE
                self.assertLess(abs(s * s + c * c - 1), Decimal("0.0000002"))


class TestReference(unittest.TestCase):
    def test_reference_known_values(self):
        # 参考自身的自洽性：sin(0)=0、cos(0)=1、sin(pi/2)≈1
        s0, c0 = cq.reference_sin_cos(0)
        self.assertEqual(s0, 0)
        self.assertEqual(c0, 1)
        s1, _ = cq.reference_sin_cos(cq.HALF_PI_Q)
        self.assertLess(abs(s1 - 1), Decimal("1e-7"))

    def test_reference_does_not_use_core_algorithm(self):
        # 独立参考模块不得导入核心的迭代实现
        import cordic_q28.reference as ref

        self.assertFalse(hasattr(ref, "sin_cos"))
        src_names = [n for n in dir(ref) if "cordic" in n.lower()]
        self.assertEqual(src_names, [])


if __name__ == "__main__":
    unittest.main()

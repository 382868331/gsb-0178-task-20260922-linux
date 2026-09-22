"""独立高精度参考：Decimal 泰勒级数 + 范围缩减。

本模块是“相对高精度参考”的来源，用于核验定点核心的误差。
它独立实现：不调用 cordic_q28.core 的任何算法函数（仅复用输入校验），
期望值完全由 Decimal 级数得出，与被测核心无共享代码路径。

参考语义：对 angle_q 所表示的精确有理数 angle_q / 2**28（弧度），
用 80 位 pi 做范围缩减后以 70 位精度泰勒级数求 sin/cos，
结果误差远低于 2**-28 的一个 ulp，可作为“真值”使用。
"""

from __future__ import annotations

from decimal import Decimal, localcontext, ROUND_FLOOR

from .core import _validate_angle
from .tables import PI, SCALE, GEN_PRECISION

REF_PRECISION = 70
# 级数截断门限：余项小于 10**-(REF_PRECISION - 8) 即停
_TAIL_EXPONENT = -(REF_PRECISION - 8)

__all__ = ["REF_PRECISION", "reference_sin_cos", "reference_sin", "reference_cos"]


def _reduce_decimal(theta: Decimal) -> Decimal:
    """把弧度角（Decimal）规约到 [-pi, pi]，使用 80 位 pi。"""
    with localcontext() as ctx:
        ctx.prec = GEN_PRECISION
        two_pi = 2 * PI
        n = ((theta + PI) / two_pi).to_integral_value(rounding=ROUND_FLOOR)
        return +(theta - n * two_pi)


def _sin_taylor(theta: Decimal) -> Decimal:
    """sin 泰勒级数，|theta| <= pi 时约 45 项收敛到 70 位。"""
    with localcontext() as ctx:
        ctx.prec = REF_PRECISION
        x2 = theta * theta
        term = theta
        total = term
        for k in range(1, 400):
            term = -term * x2 / ((2 * k) * (2 * k + 1))
            total += term
            if term == 0 or term.adjusted() < _TAIL_EXPONENT:
                break
        return +total


def _cos_taylor(theta: Decimal) -> Decimal:
    """cos 泰勒级数，|theta| <= pi。"""
    with localcontext() as ctx:
        ctx.prec = REF_PRECISION
        x2 = theta * theta
        term = Decimal(1)
        total = term
        for k in range(1, 400):
            term = -term * x2 / ((2 * k - 1) * (2 * k))
            total += term
            if term == 0 or term.adjusted() < _TAIL_EXPONENT:
                break
        return +total


def reference_sin_cos(angle_q: int) -> tuple[Decimal, Decimal]:
    """返回 (sin, cos) 的 Decimal 高精度参考值（输入为 Q4.28 原始整数）。"""
    angle_q = _validate_angle(angle_q)
    with localcontext() as ctx:
        ctx.prec = GEN_PRECISION
        theta = +(Decimal(angle_q) / SCALE)  # 精确：分母是 2 的幂
    r = _reduce_decimal(theta)
    return _sin_taylor(r), _cos_taylor(r)


def reference_sin(angle_q: int) -> Decimal:
    return reference_sin_cos(angle_q)[0]


def reference_cos(angle_q: int) -> Decimal:
    return reference_sin_cos(angle_q)[1]

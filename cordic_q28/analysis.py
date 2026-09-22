"""误差分析：把定点核心结果与独立高精度参考对比。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext

from .core import CordicResult, sin_cos
from .reference import REF_PRECISION, reference_sin_cos
from .tables import SCALE

__all__ = ["Evaluation", "evaluate"]


@dataclass(frozen=True)
class Evaluation:
    """一次定点计算与高精度参考的对照。

    误差口径：error_lsb = 定点原始整数 - 参考值 * 2**28，
    即以 Q4.28 的 1 个 ulp（2**-28）为单位的带符号误差（Decimal）。
    """

    result: CordicResult
    sin_ref: Decimal
    cos_ref: Decimal
    sin_error_lsb: Decimal
    cos_error_lsb: Decimal


def evaluate(angle_q: int) -> Evaluation:
    """计算 sin_cos 并给出相对独立参考的误差（单位：Q4.28 ulp）。"""
    result = sin_cos(angle_q)
    sin_ref, cos_ref = reference_sin_cos(angle_q)
    with localcontext() as ctx:
        ctx.prec = REF_PRECISION
        sin_err = +(Decimal(result.sin_q) - sin_ref * SCALE)
        cos_err = +(Decimal(result.cos_q) - cos_ref * SCALE)
    return Evaluation(
        result=result,
        sin_ref=sin_ref,
        cos_ref=cos_ref,
        sin_error_lsb=sin_err,
        cos_error_lsb=cos_err,
    )

"""旋转模式 CORDIC 核心：signed Q4.28，32 轮，整数域实现。

契约（公开接口，稳定）：
- sin_cos(angle_q) -> CordicResult
- angle_q 为 signed Q4.28 原始整数（弧度），范围 [-2**31, 2**31 - 1]，
  即 [-16, 16 - 2**-28) rad；超出报 AngleRangeError。
- 拒绝 bool / float（含 NaN、Infinity）/ 其他非 int 类型，报 InvalidInputError。
- 右移一律为算术右移（向负无穷取整，与 Python int 的 >> 一致）。
- 每轮迭代后及最终结果都做 signed32 溢出检查，溢出报 CordicOverflowError。
- 角度先规约到 [-pi/2, pi/2] 并保留象限符号，规约使用量化后的
  pi/2pi 常量（定点设计的一部分，见 README 已知限制）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from .errors import AngleRangeError, CordicOverflowError, InvalidInputError
from .tables import (
    ATAN_TABLE,
    FRAC_BITS,
    HALF_PI_Q,
    ITERATIONS,
    K_INV_Q,
    PI_Q,
    SCALE,
    TWO_PI_Q,
)

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1

__all__ = [
    "INT32_MIN",
    "INT32_MAX",
    "RoundState",
    "CordicResult",
    "reduce_angle",
    "sin_cos",
]


@dataclass(frozen=True)
class RoundState:
    """单轮迭代之后的状态（均为 Q4.28 原始整数）。"""

    iteration: int  # 轮次 i，0..31
    direction: int  # 本轮旋转方向 d：+1 或 -1
    x: int
    y: int
    z: int  # 剩余角度


@dataclass(frozen=True)
class CordicResult:
    """sin_cos 的完整输出。"""

    angle_q: int  # 原始输入（Q4.28）
    reduced_q: int  # 规约到 [-pi/2, pi/2] 后的角度（Q4.28）
    quadrant_sign: int  # 象限符号：+1 或 -1
    sin_q: int  # sin(angle) 的 Q4.28 原始整数
    cos_q: int  # cos(angle) 的 Q4.28 原始整数
    trace: Tuple[RoundState, ...]  # 逐轮状态，长度 ITERATIONS


def _check_int32(value: int, what: str) -> int:
    if value < INT32_MIN or value > INT32_MAX:
        raise CordicOverflowError(
            f"{what} = {value} 超出 signed32 范围 [{INT32_MIN}, {INT32_MAX}]"
        )
    return value


def _validate_angle(angle_q: object) -> int:
    """输入校验：只接受 signed32 范围内的 int；拒绝 bool 与一切 float。"""
    if isinstance(angle_q, bool):
        raise InvalidInputError("angle_q 是整数参数，拒绝 bool（True/False）")
    if isinstance(angle_q, float):
        if math.isnan(angle_q):
            raise InvalidInputError("angle_q 拒绝 NaN")
        if math.isinf(angle_q):
            raise InvalidInputError("angle_q 拒绝 Infinity")
        raise InvalidInputError(
            f"angle_q 必须是 Q4.28 原始整数（int），不接受 float 值 {angle_q!r}"
        )
    if not isinstance(angle_q, int):
        raise InvalidInputError(
            f"angle_q 必须是 int，得到类型 {type(angle_q).__name__}"
        )
    if angle_q < INT32_MIN or angle_q > INT32_MAX:
        raise AngleRangeError(
            f"angle_q = {angle_q} 超出 signed32 Q4.28 范围 "
            f"[{INT32_MIN}, {INT32_MAX}]（约 [-16, 16) rad）"
        )
    return angle_q


def _ashr(value: int, shift: int) -> int:
    """算术右移：向负无穷取整。Python 的 >> 对负整数即如此，这里显式命名以固定语义。"""
    return value >> shift


def reduce_angle(angle_q: int) -> tuple[int, int]:
    """把 Q4.28 角度规约到 [-pi/2, pi/2]，返回 (reduced_q, quadrant_sign)。

    先在整数域按 2*pi 的量化值取模到 [-pi, pi)，再折叠象限；
    sin(angle) = sign * sin(reduced)，cos(angle) = sign * cos(reduced)。
    """
    angle_q = _validate_angle(angle_q)
    # floor 除法，使余数落在 [-PI_Q, PI_Q)
    n = (angle_q + PI_Q) // TWO_PI_Q
    r = angle_q - n * TWO_PI_Q
    sign = 1
    if r > HALF_PI_Q:
        r -= PI_Q
        sign = -1
    elif r < -HALF_PI_Q:
        r += PI_Q
        sign = -1
    return r, sign


def sin_cos(angle_q: int) -> CordicResult:
    """旋转模式 CORDIC 计算 sin/cos，输入输出均为 signed Q4.28 原始整数。"""
    angle_q = _validate_angle(angle_q)
    reduced, sign = reduce_angle(angle_q)

    x = _check_int32(K_INV_Q, "x[0] (增益逆积)")
    y = 0
    z = _check_int32(reduced, "z[0] (规约角度)")
    trace = []
    for i in range(ITERATIONS):
        d = 1 if z >= 0 else -1
        # 注意：y 的更新必须使用本轮之前的 x
        x_new = _check_int32(x - d * _ashr(y, i), f"x[{i + 1}]")
        y_new = _check_int32(y + d * _ashr(x, i), f"y[{i + 1}]")
        z_new = _check_int32(z - d * ATAN_TABLE[i], f"z[{i + 1}]")
        x, y, z = x_new, y_new, z_new
        trace.append(RoundState(iteration=i, direction=d, x=x, y=y, z=z))

    return CordicResult(
        angle_q=angle_q,
        reduced_q=reduced,
        quadrant_sign=sign,
        sin_q=_check_int32(sign * y, "sin_q"),
        cos_q=_check_int32(sign * x, "cos_q"),
        trace=tuple(trace),
    )

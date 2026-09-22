"""定点 CORDIC 旋转的范围与误差核验（signed Q4.28，32 轮旋转模式）。

公开接口：
- sin_cos(angle_q) -> CordicResult     定点核心（纯整数，无浮点）
- evaluate(angle_q) -> Evaluation      核心结果 + 独立高精度参考 + 误差（LSB）
- reference_sin_cos(angle_q)           独立 Decimal 参考（不调用核心算法）
- reduce_angle(angle_q)                角度规约到 [-pi/2, pi/2] + 象限符号
- 常量：SCALE, FRAC_BITS, ITERATIONS, ATAN_TABLE, K_INV_Q, PI_Q, TWO_PI_Q,
  HALF_PI_Q, INT32_MIN, INT32_MAX
- 错误：CordicError, InvalidInputError, AngleRangeError, CordicOverflowError
"""

from .analysis import Evaluation, evaluate
from .core import (
    INT32_MAX,
    INT32_MIN,
    CordicResult,
    RoundState,
    reduce_angle,
    sin_cos,
)
from .errors import (
    AngleRangeError,
    CordicError,
    CordicOverflowError,
    InvalidInputError,
)
from .reference import REF_PRECISION, reference_cos, reference_sin, reference_sin_cos
from .tables import (
    ATAN_TABLE,
    FRAC_BITS,
    HALF_PI_Q,
    ITERATIONS,
    K_INV_Q,
    PI,
    PI_Q,
    SCALE,
    TWO_PI_Q,
)

__all__ = [
    "sin_cos",
    "evaluate",
    "reduce_angle",
    "reference_sin_cos",
    "reference_sin",
    "reference_cos",
    "CordicResult",
    "RoundState",
    "Evaluation",
    "CordicError",
    "InvalidInputError",
    "AngleRangeError",
    "CordicOverflowError",
    "SCALE",
    "FRAC_BITS",
    "ITERATIONS",
    "ATAN_TABLE",
    "K_INV_Q",
    "PI",
    "PI_Q",
    "TWO_PI_Q",
    "HALF_PI_Q",
    "INT32_MIN",
    "INT32_MAX",
    "REF_PRECISION",
]

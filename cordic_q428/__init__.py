"""Fixed-point CORDIC sin/cos and atan2/magnitude with error verification (Q4.28).

Public API:

* :func:`sin_cos` — 32-step rotation-mode CORDIC on a Q4.28 integer angle.
* :func:`sin_cos_radians` — same, from a finite float angle in radians.
* :func:`polar` — 32-step vectoring-mode CORDIC: atan2 and magnitude of a
  Q4.28 integer vector ``(x, y)``.
* :func:`polar_xy` — same, from two finite float coordinates.
* :func:`sin_cos_ref`, :func:`polar_ref` — independent Decimal reference
  oracles (Taylor series / Decimal atan + sqrt), returning Decimals.
* :class:`CordicResult`, :class:`PolarResult`, :class:`RoundState` —
  result/state data structures.
* Exceptions: :class:`CordicError` (base), :class:`FixedPointOverflowError`,
  :class:`InvalidArgumentTypeError`, :class:`NonFiniteValueError`,
  :class:`RangeReductionError`.
* Constants: ``SCALE``, ``FRAC_BITS``, ``INT32_MIN``/``INT32_MAX``,
  ``ATAN_TABLE_Q``, ``GAIN_INV_Q``, ``PI_Q``, ``HALF_PI_Q``, ``ITERATIONS``.
"""

from .cordic import (
    CordicResult,
    PolarResult,
    RoundState,
    polar,
    polar_xy,
    sin_cos,
    sin_cos_radians,
)
from .errors import (
    CordicError,
    FixedPointOverflowError,
    InvalidArgumentTypeError,
    NonFiniteValueError,
    RangeReductionError,
)
from .fixedpoint import (
    FRAC_BITS,
    INT32_MAX,
    INT32_MIN,
    MAX_REAL,
    SCALE,
    float_to_q,
    mul_q,
    q_to_float,
    sar,
)
from .reference import polar_ref, sin_cos_ref
from .tables import (
    ATAN_TABLE_Q,
    GAIN_INV_Q,
    GAIN_PREFIX_INV_Q,
    HALF_PI_Q,
    ITERATIONS,
    PI_DEC,
    PI_Q,
    TWO_PI_Q,
)

__all__ = [
    "sin_cos",
    "sin_cos_radians",
    "polar",
    "polar_xy",
    "sin_cos_ref",
    "polar_ref",
    "CordicResult",
    "PolarResult",
    "RoundState",
    "CordicError",
    "FixedPointOverflowError",
    "InvalidArgumentTypeError",
    "NonFiniteValueError",
    "RangeReductionError",
    "FRAC_BITS",
    "SCALE",
    "INT32_MIN",
    "INT32_MAX",
    "MAX_REAL",
    "float_to_q",
    "mul_q",
    "q_to_float",
    "sar",
    "ATAN_TABLE_Q",
    "GAIN_INV_Q",
    "GAIN_PREFIX_INV_Q",
    "PI_Q",
    "HALF_PI_Q",
    "TWO_PI_Q",
    "PI_DEC",
    "ITERATIONS",
]

__version__ = "1.1.0"

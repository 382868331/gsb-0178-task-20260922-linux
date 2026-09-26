"""Fixed-point CORDIC sin/cos with range and error verification (Q4.28).

Public API:

* :func:`sin_cos` — 32-step rotation-mode CORDIC on a Q4.28 integer angle.
* :func:`sin_cos_radians` — same, from a finite float angle in radians.
* :func:`sin_cos_ref` — independent Decimal Taylor reference (oracle).
* :class:`CordicResult`, :class:`RoundState` — result/state data structures.
* Exceptions: :class:`CordicError` (base), :class:`FixedPointOverflowError`,
  :class:`InvalidArgumentTypeError`, :class:`NonFiniteValueError`,
  :class:`RangeReductionError`.
* Constants: ``SCALE``, ``FRAC_BITS``, ``INT32_MIN``/``INT32_MAX``,
  ``ATAN_TABLE_Q``, ``GAIN_INV_Q``, ``PI_Q``, ``HALF_PI_Q``, ``ITERATIONS``.
"""

from .cordic import (
    HYPER_MAX_Z_Q,
    CordicResult,
    HyperCordicResult,
    HyperRoundState,
    RoundState,
    sin_cos,
    sin_cos_radians,
    sinh_cosh,
    sinh_cosh_radians,
)
from .errors import (
    CordicError,
    FixedPointOverflowError,
    InputRangeError,
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
    q_to_float,
    sar,
)
from .reference import sin_cos_ref
from .reference import sinh_cosh_ref
from .tables import (
    ATAN_TABLE_Q,
    ATANH_TABLE_Q,
    GAIN_HYP_INV_Q,
    GAIN_INV_Q,
    HALF_PI_Q,
    HYPER_ITERATIONS,
    HYPER_REPEAT_INDICES,
    ITERATIONS,
    PI_DEC,
    PI_Q,
    TWO_PI_Q,
)

__all__ = [
    "sin_cos",
    "sin_cos_radians",
    "sin_cos_ref",
    "sinh_cosh",
    "sinh_cosh_radians",
    "sinh_cosh_ref",
    "CordicResult",
    "RoundState",
    "HyperCordicResult",
    "HyperRoundState",
    "CordicError",
    "FixedPointOverflowError",
    "InputRangeError",
    "InvalidArgumentTypeError",
    "NonFiniteValueError",
    "RangeReductionError",
    "FRAC_BITS",
    "SCALE",
    "INT32_MIN",
    "INT32_MAX",
    "MAX_REAL",
    "float_to_q",
    "q_to_float",
    "sar",
    "ATAN_TABLE_Q",
    "ATANH_TABLE_Q",
    "GAIN_INV_Q",
    "GAIN_HYP_INV_Q",
    "HYPER_ITERATIONS",
    "HYPER_REPEAT_INDICES",
    "HYPER_MAX_Z_Q",
    "PI_Q",
    "HALF_PI_Q",
    "TWO_PI_Q",
    "PI_DEC",
    "ITERATIONS",
]

__version__ = "1.0.0"

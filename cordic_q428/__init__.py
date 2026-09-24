"""Fixed-point CORDIC: rotation-mode sin/cos and vectoring-mode atan2/magnitude.

Public API:

* :func:`sin_cos` — 32-step rotation-mode CORDIC on a Q4.28 integer angle.
* :func:`sin_cos_radians` — same, from a finite float angle in radians.
* :func:`atan2_mag` — 32-step vectoring-mode CORDIC: angle and magnitude of
  a Q4.28 vector ``(y, x)``.
* :func:`atan2`, :func:`magnitude` — thin wrappers returning one Q4.28 int.
* :func:`atan2_mag_radians` — vectoring convenience wrapper from floats.
* :func:`sin_cos_ref`, :func:`atan2_mag_ref` — independent Decimal
  references (oracles).
* :class:`CordicResult`, :class:`Atan2Result`, :class:`RoundState` —
  result/state data structures.
* Exceptions: :class:`CordicError` (base), :class:`FixedPointOverflowError`,
  :class:`InvalidArgumentTypeError`, :class:`NonFiniteValueError`,
  :class:`RangeReductionError`.
* Constants: ``SCALE``, ``FRAC_BITS``, ``INT32_MIN``/``INT32_MAX``,
  ``ATAN_TABLE_Q``, ``GAIN_INV_Q``, ``PI_Q``, ``HALF_PI_Q``, ``ITERATIONS``.
"""

from .cordic import (
    Atan2Result,
    CordicResult,
    RoundState,
    atan2,
    atan2_mag,
    atan2_mag_radians,
    magnitude,
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
    rshift_even,
    sar,
    saturate_int32,
)
from .reference import atan2_mag_ref, sin_cos_ref
from .tables import ATAN_TABLE_Q, GAIN_INV_Q, HALF_PI_Q, ITERATIONS, PI_DEC, PI_Q, TWO_PI_Q

__all__ = [
    "sin_cos",
    "sin_cos_radians",
    "atan2_mag",
    "atan2",
    "magnitude",
    "atan2_mag_radians",
    "sin_cos_ref",
    "atan2_mag_ref",
    "CordicResult",
    "Atan2Result",
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
    "q_to_float",
    "mul_q",
    "rshift_even",
    "sar",
    "saturate_int32",
    "ATAN_TABLE_Q",
    "GAIN_INV_Q",
    "PI_Q",
    "HALF_PI_Q",
    "TWO_PI_Q",
    "PI_DEC",
    "ITERATIONS",
]

__version__ = "1.1.0"

"""Fixed-point CORDIC sin/cos with range and error verification (Q4.28).

Public API:

* :func:`sin_cos` — 32-step rotation-mode CORDIC on a Q4.28 integer angle
  (``with_envelope=True`` attaches the conservative error envelope).
* :func:`sin_cos_radians` — same, from a finite float angle in radians.
* :func:`sin_cos_ref` — independent Decimal Taylor reference (oracle).
* :func:`compute_envelope` — standalone conservative error-envelope proof.
* :class:`CordicResult`, :class:`RoundState` — result/state data structures.
* :class:`EnvelopeResult`, :class:`RoundEnvelope`, :class:`ErrorSources`,
  :class:`SourceWidth` — envelope proof data structures.
* Exceptions: :class:`CordicError` (base), :class:`FixedPointOverflowError`,
  :class:`InvalidArgumentTypeError`, :class:`NonFiniteValueError`,
  :class:`RangeReductionError`.
* Constants: ``SCALE``, ``FRAC_BITS``, ``INT32_MIN``/``INT32_MAX``,
  ``ATAN_TABLE_Q``, ``GAIN_INV_Q``, ``PI_Q``, ``HALF_PI_Q``, ``ITERATIONS``.
"""

from .cordic import CordicResult, RoundState, sin_cos, sin_cos_radians
from .envelope import (
    ErrorSources,
    EnvelopeResult,
    RoundEnvelope,
    SourceWidth,
    compute_envelope,
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
    q_to_float,
    sar,
)
from .reference import sin_cos_ref
from .tables import ATAN_TABLE_Q, GAIN_INV_Q, HALF_PI_Q, ITERATIONS, PI_DEC, PI_Q, TWO_PI_Q

__all__ = [
    "sin_cos",
    "sin_cos_radians",
    "sin_cos_ref",
    "compute_envelope",
    "CordicResult",
    "RoundState",
    "EnvelopeResult",
    "RoundEnvelope",
    "ErrorSources",
    "SourceWidth",
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
    "sar",
    "ATAN_TABLE_Q",
    "GAIN_INV_Q",
    "PI_Q",
    "HALF_PI_Q",
    "TWO_PI_Q",
    "PI_DEC",
    "ITERATIONS",
]

__version__ = "1.0.0"

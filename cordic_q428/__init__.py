"""Fixed-point CORDIC sin/cos with range and error verification (Q4.28).

Public API:

* :func:`sin_cos` — 32-step rotation-mode CORDIC on a Q4.28 integer angle;
  pass ``with_envelope=True`` to attach a proven integer error envelope.
* :func:`sin_cos_with_envelope` — named convenience for the envelope form.
* :func:`sin_cos_radians` — same, from a finite float angle in radians.
* :func:`sin_cos_ref` — independent Decimal Taylor reference (oracle).
* :class:`CordicResult`, :class:`RoundState` — result/state data structures.
* :class:`SinCosEnvelope`, :class:`EnvelopeSources`, :class:`RoundEnvelope`,
  :class:`IntervalBounds` — envelope proof structures
  (:mod:`cordic_q428.envelope`).
* Exceptions: :class:`CordicError` (base), :class:`FixedPointOverflowError`,
  :class:`InvalidArgumentTypeError`, :class:`NonFiniteValueError`,
  :class:`RangeReductionError`.
* Constants: ``SCALE``, ``FRAC_BITS``, ``INT32_MIN``/``INT32_MAX``,
  ``ATAN_TABLE_Q``, ``GAIN_INV_Q``, ``PI_Q``, ``HALF_PI_Q``, ``ITERATIONS``.
"""

from .cordic import CordicResult, RoundState, sin_cos, sin_cos_radians, sin_cos_with_envelope
from .envelope import (
    EnvelopeSources,
    IntervalBounds,
    RoundEnvelope,
    SinCosEnvelope,
    envelope_for_reduced,
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
from .tables import (
    ATAN_TABLE_DEC,
    ATAN_TABLE_Q,
    GAIN_INV_DEC,
    GAIN_INV_Q,
    HALF_PI_Q,
    ITERATIONS,
    PI_DEC,
    PI_Q,
    TWO_PI_Q,
)

__all__ = [
    "sin_cos",
    "sin_cos_with_envelope",
    "sin_cos_radians",
    "sin_cos_ref",
    "CordicResult",
    "RoundState",
    "SinCosEnvelope",
    "EnvelopeSources",
    "RoundEnvelope",
    "IntervalBounds",
    "envelope_for_reduced",
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
    "ATAN_TABLE_DEC",
    "GAIN_INV_Q",
    "GAIN_INV_DEC",
    "PI_Q",
    "HALF_PI_Q",
    "TWO_PI_Q",
    "PI_DEC",
    "ITERATIONS",
]

__version__ = "1.1.0"

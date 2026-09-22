"""Signed Q4.28 fixed-point primitives.

Q4.28 stores a real value ``v`` as the integer ``round(v * 2**28)`` in a
signed 32-bit word.  Representable range is ``[-8.0, 8.0 - 2**-28]`` and the
resolution (one ulp) is ``2**-28``.  All arithmetic in this module is exact
integer arithmetic; the right shift is defined as *arithmetic shift rounding
toward negative infinity* (Python's ``>>`` on ``int`` has exactly this
semantics, and :func:`sar` makes the contract explicit and testable).
"""

from __future__ import annotations

from .errors import FixedPointOverflowError, InvalidArgumentTypeError, NonFiniteValueError

FRAC_BITS: int = 28
SCALE: int = 1 << FRAC_BITS
INT32_MIN: int = -(1 << 31)
INT32_MAX: int = (1 << 31) - 1
# Largest real value representable in Q4.28 (INT32_MAX / SCALE).
MAX_REAL: float = INT32_MAX / SCALE


def require_int(value: object, name: str) -> int:
    """Return ``value`` if it is a genuine ``int``; reject ``bool`` and others.

    ``bool`` is a subclass of ``int`` in Python, so it is checked first —
    passing ``True`` as a fixed-point sample is a caller bug, not data.
    """
    if isinstance(value, bool):
        raise InvalidArgumentTypeError(
            f"{name}: bool is not a valid fixed-point integer (got {value!r})"
        )
    if not isinstance(value, int):
        raise InvalidArgumentTypeError(
            f"{name}: expected int, got {type(value).__name__} ({value!r})"
        )
    return value


def require_finite_real(value: object, name: str) -> float:
    """Return ``float(value)`` for a finite int/float; reject NaN/Inf and bool."""
    if isinstance(value, bool):
        raise InvalidArgumentTypeError(
            f"{name}: bool is not a valid numeric input (got {value!r})"
        )
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        if value != value:  # NaN
            raise NonFiniteValueError(f"{name}: NaN is not a valid numeric input")
        if value in (float("inf"), float("-inf")):
            raise NonFiniteValueError(f"{name}: infinity is not a valid numeric input")
        return value
    raise InvalidArgumentTypeError(
        f"{name}: expected int or float, got {type(value).__name__} ({value!r})"
    )


def check_int32(value: int, where: str) -> int:
    """Assert ``value`` fits a signed 32-bit word; raise otherwise."""
    if value < INT32_MIN or value > INT32_MAX:
        raise FixedPointOverflowError(
            f"signed 32-bit overflow: {value} not in [{INT32_MIN}, {INT32_MAX}]",
            value=value,
            where=where,
        )
    return value


def sar(value: int, shift: int) -> int:
    """Arithmetic shift right, rounding toward negative infinity.

    ``sar(v, s) == floor(v / 2**s)`` for every integer ``v`` and ``s >= 0``.
    This is the mandated shift semantics for the CORDIC datapath.
    """
    require_int(value, "sar.value")
    require_int(shift, "sar.shift")
    if shift < 0:
        raise InvalidArgumentTypeError(f"sar.shift: negative shift {shift} is not supported")
    return value >> shift


def float_to_q(value: object, name: str = "value") -> int:
    """Convert a finite int/float to Q4.28 with round-half-to-even.

    Raises :class:`NonFiniteValueError` for NaN/Inf and
    :class:`FixedPointOverflowError` if the result leaves the signed 32-bit
    range.  Never silently clamps.
    """
    real = require_finite_real(value, name)
    scaled = real * SCALE
    q = round(scaled)  # Python round() is ties-to-even on floats
    return check_int32(q, name)


def q_to_float(q: object) -> float:
    """Convert a Q4.28 integer to the nearest ``float``."""
    q = require_int(q, "q")
    return q / SCALE

"""Independent high-precision sin/cos reference built on Decimal Taylor series.

This module is the *verification oracle* for the CORDIC core.  It shares no
code path with :mod:`cordic_q428.cordic`: the angle is reduced with Decimal
arithmetic into ``[-pi, pi]`` and then sin/cos are summed directly from their
Taylor series at 60-digit precision.  It never calls the CORDIC core, and the
core never calls it except to *report* the error against it.

Interface: :func:`sin_cos_ref` takes a Q4.28 integer angle and returns
``(sin, cos)`` as :class:`~decimal.Decimal` values.  :func:`polar_ref` takes
two Q4.28 integers ``(x, y)`` and returns ``(atan2(y, x), hypot(x, y))`` as
Decimals in real units (radians and the Q4.28 real scale), derived
independently with a Decimal atan series and a Decimal square root.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

from .errors import InvalidArgumentTypeError
from .fixedpoint import INT32_MAX, INT32_MIN, SCALE, require_int
from .tables import PI_DEC, _atan_series

_REF_PREC: int = 60
_TWO_PI_DEC = PI_DEC * 2


def _reduce_to_pi(x: Decimal) -> Decimal:
    """Reduce a Decimal angle to (-pi, pi] by exact multiples of 2*pi."""
    # Decimal % keeps the sign of the divisor for fmod-like semantics; we use
    # explicit floor division to stay exact and deterministic.
    k = (x / _TWO_PI_DEC).to_integral_value(rounding="ROUND_FLOOR")
    r = x - k * _TWO_PI_DEC
    if r > PI_DEC:
        r -= _TWO_PI_DEC
    return r


def _taylor_sin_cos(x: Decimal, prec: int) -> tuple[Decimal, Decimal]:
    """Sum the Taylor series for sin and cos at reduced angle x (|x| <= pi)."""
    with localcontext() as ctx:
        ctx.prec = prec
        x2 = x * x
        eps = Decimal(10) ** (-(prec - 8))
        # sin: term_k = (-1)^k x^(2k+1)/(2k+1)!
        sin_total = x
        sin_term = x
        # cos: term_k = (-1)^k x^(2k)/(2k)!
        cos_total = Decimal(1)
        cos_term = Decimal(1)
        k = 1
        while True:
            sin_term *= -x2 / ((2 * k) * (2 * k + 1))
            cos_term *= -x2 / ((2 * k - 1) * (2 * k))
            sin_total += sin_term
            cos_total += cos_term
            if abs(sin_term) < eps and abs(cos_term) < eps:
                return +sin_total, +cos_total
            k += 1


def sin_cos_ref(angle_q: int) -> tuple[Decimal, Decimal]:
    """Return (sin, cos) of the Q4.28 angle as high-precision Decimals.

    Independent of the CORDIC core: Decimal range reduction + Taylor series.
    """
    require_int(angle_q, "angle_q")
    if not (INT32_MIN <= angle_q <= INT32_MAX):
        raise InvalidArgumentTypeError(
            f"angle_q: {angle_q} outside signed 32-bit Q4.28 range "
            f"[{INT32_MIN}, {INT32_MAX}]"
        )
    with localcontext() as ctx:
        ctx.prec = _REF_PREC
        angle = Decimal(angle_q) / SCALE
        reduced = _reduce_to_pi(angle)
        return _taylor_sin_cos(reduced, _REF_PREC)


def _atan_decimal(t: Decimal, prec: int) -> Decimal:
    """atan(t) for t >= 0, with the Taylor argument always reduced to <= a.

    The plain atan Taylor series converges like ``arg**(2k)``; at an argument
    near 1 (a vector near a 45-degree diagonal) that needs millions of terms.
    Two angle identities keep the series argument in ``[0, a]`` where
    ``a = tan(pi/8) = sqrt(2) - 1 ~= 0.4142``:

    * ``t <= a``      -> atan(t) directly;
    * ``a < t <= 1/a`` -> pi/4 + atan((t-1)/(t+1))  (the fraction is in [-a, a]);
    * ``t > 1/a``     -> pi/2 - atan(1/t)           (``1/t < a``).
    """
    if t == 0:
        return Decimal(0)
    a = Decimal(2).sqrt() - 1  # tan(pi/8)
    if t <= a:
        return _atan_series(t, prec)
    if t <= 1 / a:
        return PI_DEC / 4 + _atan_series((t - 1) / (t + 1), prec)
    return PI_DEC / 2 - _atan_series(1 / t, prec)


def polar_ref(x_q: int, y_q: int) -> tuple[Decimal, Decimal]:
    """Return ``(atan2(y, x), hypot(x, y))`` of a Q4.28 vector, as Decimals.

    The angle is in radians (real units, range ``(-pi, pi]``; the zero vector
    reports angle 0) and the magnitude is in the Q4.28 real scale
    (``sqrt(x**2 + y**2) / 2**28``).  Independent of the CORDIC vector-mode
    core: quadrant handling by signs, a Decimal atan series, and a Decimal
    square root.
    """
    require_int(x_q, "x_q")
    require_int(y_q, "y_q")
    for name, v in (("x_q", x_q), ("y_q", y_q)):
        if not (INT32_MIN <= v <= INT32_MAX):
            raise InvalidArgumentTypeError(
                f"{name}: {v} outside signed 32-bit Q4.28 range "
                f"[{INT32_MIN}, {INT32_MAX}]"
            )
    with localcontext() as ctx:
        ctx.prec = _REF_PREC
        x = Decimal(x_q) / SCALE
        y = Decimal(y_q) / SCALE
        if x_q == 0 and y_q == 0:
            angle = Decimal(0)
        elif x_q > 0:
            angle = _atan_decimal(abs(y / x), _REF_PREC)
            if y_q < 0:
                angle = -angle
        elif x_q < 0:
            angle = PI_DEC - _atan_decimal(abs(y / x), _REF_PREC)
            if y_q < 0:
                angle = -angle  # -pi + atan(|.|), landing in (-pi, 0)
        else:  # x == 0, y != 0
            angle = PI_DEC / 2 if y_q > 0 else -PI_DEC / 2
        magnitude = (x * x + y * y).sqrt()
        return +angle, +magnitude

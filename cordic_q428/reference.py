"""Independent high-precision trigonometric references built on Decimal series.

This module is the *verification oracle* for the CORDIC core.  It shares no
code path with :mod:`cordic_q428.cordic`:

* :func:`sin_cos_ref` reduces the angle with Decimal arithmetic into
  ``[-pi, pi]`` and sums sin/cos directly from their Taylor series.
* :func:`atan2_mag_ref` computes ``atan2(y, x)`` from the atan Taylor series
  (half-angle identity used to pull the argument below 0.5) and the magnitude
  with :meth:`decimal.Decimal.sqrt`.

Both run at 60-digit precision, never call the CORDIC core, and evaluate the
*quantized* Q4.28 inputs (not idealized reals), so reported residuals are the
genuine fixed-point errors.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

from .errors import InvalidArgumentTypeError
from .fixedpoint import INT32_MAX, INT32_MIN, SCALE, require_int
from .tables import PI_DEC

_REF_PREC: int = 60
_TWO_PI_DEC = PI_DEC * 2
_HALF = Decimal(1) / 2


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


def _atan_half_pull(t: Decimal, prec: int) -> tuple[Decimal, int]:
    """Pull an atan argument toward 0 with the half-angle identity.

    ``atan(t) = 2*atan(t / (1 + sqrt(1 + t**2)))``; repeated until
    ``|t| <= 0.5`` (at most 3 applications for any representable Q4.28 ratio,
    whose magnitude is at most 2**31).  Returns the reduced argument and the
    number of applications ``n``; the final angle is the series value times
    ``2**n``.
    """
    n = 0
    while abs(t) > _HALF:
        t = t / (Decimal(1) + (Decimal(1) + t * t).sqrt())
        n += 1
    return +t, n


def _atan_ref(v: Decimal, prec: int) -> Decimal:
    """High-precision atan via half-angle pull followed by Taylor series."""
    with localcontext() as ctx:
        ctx.prec = prec
        t, n = _atan_half_pull(+v, prec)
        t2 = t * t
        term = t
        total = t
        eps = Decimal(10) ** (-(prec - 8))
        k = 1
        while True:
            term *= t2
            add = term / (2 * k + 1)
            total += -add if k % 2 else add
            if abs(add) < eps:
                return (+total) * (Decimal(2) ** n)
            k += 1


def atan2_mag_ref(y_q: int, x_q: int) -> tuple[Decimal, Decimal]:
    """Independent oracle for :func:`cordic_q428.cordic.atan2_mag`.

    Returns ``(atan2(y, x), sqrt(x**2 + y**2))`` in real units, as Decimals,
    computed solely from Decimal atan series and :meth:`Decimal.sqrt`.
    Conventions match the core exactly: the zero vector gives angle 0, and
    the negative x-axis with ``y == 0`` gives ``+pi`` (integer inputs carry
    no signed zero).
    """
    require_int(y_q, "y_q")
    require_int(x_q, "x_q")
    for name, q in (("y_q", y_q), ("x_q", x_q)):
        if not (INT32_MIN <= q <= INT32_MAX):
            raise InvalidArgumentTypeError(
                f"{name}: {q} outside signed 32-bit Q4.28 range "
                f"[{INT32_MIN}, {INT32_MAX}]"
            )
    with localcontext() as ctx:
        ctx.prec = _REF_PREC
        x = Decimal(x_q) / SCALE
        y = Decimal(y_q) / SCALE
        if x == 0 and y == 0:
            return Decimal(0), Decimal(0)
        mag = (x * x + y * y).sqrt()
        if x > 0:
            angle = _atan_ref(y / x, _REF_PREC)
        elif x < 0:
            angle = _atan_ref(y / x, _REF_PREC)
            angle += PI_DEC if y >= 0 else -PI_DEC
        else:  # x == 0, y != 0
            angle = PI_DEC / 2 if y > 0 else -PI_DEC / 2
        return +angle, +mag

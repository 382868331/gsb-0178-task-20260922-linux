"""Independent high-precision sin/cos reference built on Decimal Taylor series.

This module is the *verification oracle* for the CORDIC core.  It shares no
code path with :mod:`cordic_q428.cordic`: the angle is reduced with Decimal
arithmetic into ``[-pi, pi]`` and then sin/cos are summed directly from their
Taylor series at 60-digit precision.  It never calls the CORDIC core, and the
core never calls it except to *report* the error against it.

Interface: :func:`sin_cos_ref` takes a Q4.28 integer angle and returns
``(sin, cos)`` as :class:`~decimal.Decimal` values.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

from .errors import InvalidArgumentTypeError
from .fixedpoint import INT32_MAX, INT32_MIN, SCALE, require_int
from .tables import PI_DEC

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


_HYP_REF_PREC: int = 60


def sinh_cosh_ref(z_q: int) -> tuple[Decimal, Decimal]:
    """Return (sinh, cosh) of the Q4.28 angle z as high-precision Decimals.

    Independent oracle for the hyperbolic CORDIC core, built directly on the
    exponential definition with Decimal arithmetic only:

        cosh(z) = (e^z + e^-z) / 2,   sinh(z) = (e^z - e^-z) / 2

    It shares no code path with the hyperbolic CORDIC core and never calls
    it.  |z| <= 1 is the contracted domain of the core; the reference itself
    accepts any signed 32-bit Q4.28 input.
    """
    require_int(z_q, "z_q")
    if not (INT32_MIN <= z_q <= INT32_MAX):
        raise InvalidArgumentTypeError(
            f"z_q: {z_q} outside signed 32-bit Q4.28 range "
            f"[{INT32_MIN}, {INT32_MAX}]"
        )
    with localcontext() as ctx:
        ctx.prec = _HYP_REF_PREC
        z = Decimal(z_q) / SCALE
        e_pos = z.exp()
        e_neg = (-z).exp()
        two = Decimal(2)
        sinh = (e_pos - e_neg) / two
        cosh = (e_pos + e_neg) / two
        return +sinh, +cosh

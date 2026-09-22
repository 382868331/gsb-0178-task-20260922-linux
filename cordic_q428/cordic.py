"""32-step rotation-mode CORDIC in signed Q4.28 fixed point.

Public interface
----------------
``sin_cos(angle_q) -> CordicResult``
    ``angle_q`` is a signed 32-bit integer holding a Q4.28 radian angle in
    ``[-8.0, 8.0)``.  Returns sin/cos as Q4.28 integers, the per-round state
    trace, and the error against the independent Decimal reference.

Contract highlights (see README for the full contract):

* ``bool`` is rejected as an integer argument; NaN/Infinity are rejected as
  real-valued arguments (see :func:`sin_cos_radians`).
* The angle is first reduced into ``[-pi/2, pi/2]`` by adding/subtracting the
  quantized pi, recording the quadrant sign flip, then a 32-iteration
  rotation-mode CORDIC runs on the reduced angle.
* Every right shift is arithmetic, rounding toward negative infinity
  (:func:`cordic_q428.fixedpoint.sar`).
* Every register update is range-checked: any value leaving the signed
  32-bit range raises :class:`FixedPointOverflowError`.  Nothing wraps.
* ``math.sin``/``math.cos`` are never used anywhere in this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext

from .errors import FixedPointOverflowError, RangeReductionError
from .fixedpoint import (
    INT32_MAX,
    INT32_MIN,
    SCALE,
    check_int32,
    float_to_q,
    require_int,
)
from .reference import sin_cos_ref
from .tables import ATAN_TABLE_Q, GAIN_INV_Q, HALF_PI_Q, ITERATIONS, PI_Q

# The reduction loop below needs at most ceil(8 / pi) = 3 steps for any
# representable Q4.28 angle; the budget is a defensive bound, and exhausting
# it is reported separately from ordinary overflow.
_REDUCTION_BUDGET: int = 8


@dataclass(frozen=True)
class RoundState:
    """State of the CORDIC datapath after one iteration.

    All of ``x``, ``y``, ``z`` are signed Q4.28 integers; ``direction`` is
    +1 or -1 (the sign of ``z`` *before* this iteration's update).
    """

    index: int
    direction: int
    x: int
    y: int
    z: int


@dataclass(frozen=True)
class CordicResult:
    """Result of :func:`sin_cos`.

    ``sin_q``/``cos_q`` are the Q4.28 integer outputs.  ``rounds`` holds the
    32 per-iteration states.  ``sin_error``/``cos_error`` are Decimals:
    ``output/2**28 - reference`` in real units (signed; the reference is the
    independent Decimal Taylor oracle in :mod:`cordic_q428.reference`).
    ``reduced_q`` is the angle after quadrant reduction and ``sign_flip`` is
    +1 or -1, the quadrant sign that was factored out and reapplied.
    """

    sin_q: int
    cos_q: int
    reduced_q: int
    sign_flip: int
    rounds: tuple[RoundState, ...]
    sin_error: Decimal
    cos_error: Decimal

    @property
    def sin(self) -> float:
        return self.sin_q / SCALE

    @property
    def cos(self) -> float:
        return self.cos_q / SCALE


def _validate_angle_q(angle_q: object) -> int:
    angle_q = require_int(angle_q, "angle_q")
    if not (INT32_MIN <= angle_q <= INT32_MAX):
        raise FixedPointOverflowError(
            f"angle_q: {angle_q} outside signed 32-bit Q4.28 range "
            f"[{INT32_MIN}, {INT32_MAX}] (radians must lie in [-8.0, 8.0))",
            value=angle_q,
            where="angle_q",
        )
    return angle_q


def _reduce_to_half_pi(angle_q: int) -> tuple[int, int]:
    """Reduce to [-pi/2, pi/2]; return (reduced_q, sign_flip).

    Subtracting pi rotates the angle by half a turn, which negates both sin
    and cos; each subtraction therefore flips ``sign_flip``.  All arithmetic
    is on checked signed 32-bit Q4.28 integers.
    """
    reduced = angle_q
    sign = 1
    steps = 0
    while reduced > HALF_PI_Q or reduced < -HALF_PI_Q:
        if steps >= _REDUCTION_BUDGET:
            raise RangeReductionError(
                f"quadrant reduction did not converge within {_REDUCTION_BUDGET} "
                f"steps for angle_q={angle_q}"
            )
        if reduced > HALF_PI_Q:
            reduced = check_int32(reduced - PI_Q, "reduction.sub_pi")
        else:
            reduced = check_int32(reduced + PI_Q, "reduction.add_pi")
        sign = -sign
        steps += 1
    return reduced, sign


def sin_cos(angle_q: int) -> CordicResult:
    """Compute sin and cos of a Q4.28 angle with 32-step rotation CORDIC."""
    angle_q = _validate_angle_q(angle_q)
    reduced, sign_flip = _reduce_to_half_pi(angle_q)

    x = GAIN_INV_Q
    y = 0
    z = reduced
    rounds: list[RoundState] = []
    for i in range(ITERATIONS):
        d = 1 if z >= 0 else -1
        # Arithmetic shift right, rounding toward -inf, on the *old* values.
        x_shift = x >> i
        y_shift = y >> i
        if d > 0:
            x_new = check_int32(x - y_shift, f"round[{i}].x")
            y_new = check_int32(y + x_shift, f"round[{i}].y")
            z_new = check_int32(z - ATAN_TABLE_Q[i], f"round[{i}].z")
        else:
            x_new = check_int32(x + y_shift, f"round[{i}].x")
            y_new = check_int32(y - x_shift, f"round[{i}].y")
            z_new = check_int32(z + ATAN_TABLE_Q[i], f"round[{i}].z")
        x, y, z = x_new, y_new, z_new
        rounds.append(RoundState(index=i, direction=d, x=x, y=y, z=z))

    cos_q = check_int32(sign_flip * x, "output.cos")
    sin_q = check_int32(sign_flip * y, "output.sin")

    ref_sin, ref_cos = sin_cos_ref(angle_q)
    with localcontext() as ctx:
        ctx.prec = 50
        sin_err = Decimal(sin_q) / SCALE - ref_sin
        cos_err = Decimal(cos_q) / SCALE - ref_cos

    return CordicResult(
        sin_q=sin_q,
        cos_q=cos_q,
        reduced_q=reduced,
        sign_flip=sign_flip,
        rounds=tuple(rounds),
        sin_error=+sin_err,
        cos_error=+cos_err,
    )


def sin_cos_radians(angle: float) -> CordicResult:
    """Convenience wrapper: convert a finite float radian angle to Q4.28 first.

    NaN and +/-infinity are rejected (:class:`NonFiniteValueError`); values
    outside ``[-8.0, 8.0)`` raise :class:`FixedPointOverflowError` rather than
    being silently wrapped.
    """
    return sin_cos(float_to_q(angle, "angle"))

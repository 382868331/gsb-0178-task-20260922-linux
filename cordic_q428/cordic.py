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
from .errors import InputRangeError
from .reference import sin_cos_ref, sinh_cosh_ref
from .tables import (
    ATAN_TABLE_Q,
    ATANH_TABLE_Q,
    GAIN_HYP_INV_Q,
    GAIN_INV_Q,
    HALF_PI_Q,
    HYPER_ITERATIONS,
    HYPER_REPEAT_INDICES,
    ITERATIONS,
    PI_Q,
)

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


# ---------------------------------------------------------------------------
# Hyperbolic rotation mode: cosh/sinh.
# ---------------------------------------------------------------------------

#: Largest |z| (in Q4.28) accepted by :func:'sinh_cosh': exactly 1.0.
HYPER_MAX_Z_Q: int = SCALE


@dataclass(frozen=True)
class HyperRoundState:
    """State of the hyperbolic datapath after one micro-rotation.

    'index' is the CORDIC iteration i (1-based); 'repeated' is True on the
    extra execution of i = 4, 13, 40 (i_next = 3*i + 1).  'shift' is the
    actual shift amount applied (equal to 'index'); 'x_shift'/'y_shift' are
    the actual shifted addends 'x >> shift'/'y >> shift' used this round —
    i.e. the exact shift-and-round (toward -inf) record of the round.
    'direction' is +1/-1 (sign of z before the update); 'x', 'y', 'z' are
    the signed Q4.28 registers after the update.
    """

    index: int
    repeated: bool
    direction: int
    shift: int
    x_shift: int
    y_shift: int
    x: int
    y: int
    z: int


@dataclass(frozen=True)
class HyperCordicResult:
    """Result of :func:'sinh_cosh'.

    'sinh_q'/'cosh_q' are the Q4.28 integer outputs.  'rounds' holds every
    executed micro-rotation (HYPER_ITERATIONS + len(HYPER_REPEAT_INDICES)
    entries, repeats included).  'sinh_error'/'cosh_error' are Decimals:
    'output/2**28 - reference' in real units, against the independent
    Decimal exponential oracle in :mod:'cordic_q428.reference'.
    """

    sinh_q: int
    cosh_q: int
    rounds: tuple[HyperRoundState, ...]
    sinh_error: Decimal
    cosh_error: Decimal

    @property
    def sinh(self) -> float:
        return self.sinh_q / SCALE

    @property
    def cosh(self) -> float:
        return self.cosh_q / SCALE


def _validate_z_q(z_q: object) -> int:
    z_q = require_int(z_q, "z_q")
    if not (INT32_MIN <= z_q <= INT32_MAX):
        raise FixedPointOverflowError(
            f"z_q: {z_q} outside signed 32-bit Q4.28 range "
            f"[{INT32_MIN}, {INT32_MAX}]",
            value=z_q,
            where="z_q",
        )
    if not (-HYPER_MAX_Z_Q <= z_q <= HYPER_MAX_Z_Q):
        raise InputRangeError(
            f"z_q: {z_q} outside hyperbolic convergence range "
            f"[{-HYPER_MAX_Z_Q}, {HYPER_MAX_Z_Q}] (|z| <= 1.0 rad); "
            "input refused, not saturated",
            value=z_q,
            where="z_q",
        )
    return z_q


def sinh_cosh(z_q: int) -> HyperCordicResult:
    """Compute cosh(z) and sinh(z) of a Q4.28 angle with hyperbolic CORDIC.

    Rotation mode starting at i = 1, repeating the rounds at i = 4, 13, 40
    (i_next = 3*i + 1); the inverse gain constant accounts for the repeats.
    Only |z| <= 1.0 is supported; anything larger raises
    :class:'InputRangeError'.  Every register update is range-checked — any
    intermediate overflow raises :class:'FixedPointOverflowError' instead of
    saturating.
    """
    z_q = _validate_z_q(z_q)

    x = GAIN_HYP_INV_Q
    y = 0
    z = z_q
    rounds: list[HyperRoundState] = []
    for i in range(1, HYPER_ITERATIONS + 1):
        reps = 2 if i in HYPER_REPEAT_INDICES else 1
        for rep in range(reps):
            d = 1 if z >= 0 else -1
            x_shift = x >> i
            y_shift = y >> i
            tag = f"round[i={i}{' rep' if rep else ''}]"
            if d > 0:
                x_new = check_int32(x + y_shift, tag + ".x")
                y_new = check_int32(y + x_shift, tag + ".y")
                z_new = check_int32(z - ATANH_TABLE_Q[i - 1], tag + ".z")
            else:
                x_new = check_int32(x - y_shift, tag + ".x")
                y_new = check_int32(y - x_shift, tag + ".y")
                z_new = check_int32(z + ATANH_TABLE_Q[i - 1], tag + ".z")
            x, y, z = x_new, y_new, z_new
            rounds.append(
                HyperRoundState(
                    index=i,
                    repeated=bool(rep),
                    direction=d,
                    shift=i,
                    x_shift=x_shift,
                    y_shift=y_shift,
                    x=x,
                    y=y,
                    z=z,
                )
            )

    cosh_q = check_int32(x, "output.cosh")
    sinh_q = check_int32(y, "output.sinh")

    ref_sinh, ref_cosh = sinh_cosh_ref(z_q)
    with localcontext() as ctx:
        ctx.prec = 50
        sinh_err = Decimal(sinh_q) / SCALE - ref_sinh
        cosh_err = Decimal(cosh_q) / SCALE - ref_cosh

    return HyperCordicResult(
        sinh_q=sinh_q,
        cosh_q=cosh_q,
        rounds=tuple(rounds),
        sinh_error=+sinh_err,
        cosh_error=+cosh_err,
    )


def sinh_cosh_radians(z: float) -> HyperCordicResult:
    """Convenience wrapper: finite float -> Q4.28 -> :func:'sinh_cosh'.

    NaN/+/-infinity are rejected (:class:'NonFiniteValueError'); |z| > 1.0
    raises :class:'InputRangeError' rather than being clamped.
    """
    return sinh_cosh(float_to_q(z, "z"))

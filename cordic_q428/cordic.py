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
    FRAC_BITS,
    INT32_MAX,
    INT32_MIN,
    SCALE,
    check_int32,
    float_to_q,
    mul_q,
    require_int,
    rshift_even,
    saturate_int32,
)
from .reference import atan2_mag_ref, sin_cos_ref
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


@dataclass(frozen=True)
class Atan2Result:
    """Result of :func:`atan2_mag` (vectoring-mode CORDIC).

    ``angle_q`` is the Q4.28 angle ``atan2(y, x)`` in ``[-pi, pi]``;
    ``mag_q`` is the Q4.28 magnitude ``sqrt(x**2 + y**2)`` saturated to the
    signed 32-bit maximum when the true magnitude exceeds ``MAX_REAL``
    (``saturated`` is True exactly then).  ``angle_offset_q`` is the quadrant
    offset applied during preprocessing: 0, ``+PI_Q`` or ``-PI_Q``.  The two
    transverse registers are intentionally carried in arbitrary-precision
    integers across the iterations (in the saturation regime both can exceed
    the signed 32-bit ceiling; folded corners have magnitude ~sqrt(2)*8), so
    the raw ``x``/``y`` values in ``rounds`` may lie outside the signed
    32-bit range while ``z`` always stays inside it.  Only the committed
    outputs are range-checked or saturated.  ``angle_error``/``mag_error``
    are Decimals against the independent Decimal oracle
    (:func:`cordic_q428.reference.atan2_mag_ref`), in real units; for a
    saturated result ``mag_error`` is the (negative) clamp gap.
    """

    angle_q: int
    mag_q: int
    saturated: bool
    angle_offset_q: int
    rounds: tuple[RoundState, ...]
    angle_error: Decimal
    mag_error: Decimal

    @property
    def angle(self) -> float:
        return self.angle_q / SCALE

    @property
    def mag(self) -> float:
        return self.mag_q / SCALE


def _validate_coord_q(value: object, name: str) -> int:
    value = require_int(value, name)
    if not (INT32_MIN <= value <= INT32_MAX):
        raise FixedPointOverflowError(
            f"{name}: {value} outside signed 32-bit Q4.28 range "
            f"[{INT32_MIN}, {INT32_MAX}] (coordinates must lie in [-8.0, 8.0))",
            value=value,
            where=name,
        )
    return value


def atan2_mag(y_q: int, x_q: int) -> Atan2Result:
    """Vectoring-mode CORDIC: angle ``atan2(y, x)`` and magnitude.

    Arguments follow the ``atan2(y, x)`` convention (vertical coordinate
    first).  Both are signed 32-bit Q4.28 integers.  Defined results for the
    special cases:

    * zero vector ``(0, 0)`` -> angle ``0``, magnitude ``0``;
    * negative x-axis with ``y == 0`` -> angle ``+pi`` (integer inputs have no
      signed-zero distinction; documented convention);
    * coordinates containing ``INT32_MIN`` are handled in arbitrary-precision
      preprocessing (negating ``INT32_MIN`` never wraps);
    * magnitude above the Q4.28 maximum -> magnitude saturates at
      ``INT32_MAX`` with ``saturated=True``; the angle is still computed.
    """
    y_q = _validate_coord_q(y_q, "y_q")
    x_q = _validate_coord_q(x_q, "x_q")

    # --- Zero vector: angle is defined as 0 (documented convention, matching
    # math.atan2(0, 0)); no rotation is defined, so the trace is 32 explicit
    # idle states (direction 0, all registers zero).
    if x_q == 0 and y_q == 0:
        idle = tuple(RoundState(index=i, direction=0, x=0, y=0, z=0) for i in range(ITERATIONS))
        return Atan2Result(
            angle_q=0,
            mag_q=0,
            saturated=False,
            angle_offset_q=0,
            rounds=idle,
            angle_error=Decimal(0),
            mag_error=Decimal(0),
        )

    # --- Quadrant preprocessing, in arbitrary-precision integers: negating
    # INT32_MIN would overflow a 32-bit word, and the (x, y) -> (-x, -y) flip
    # is a rotation by pi, recorded as the angle offset.  y == 0 on the
    # negative x-axis takes the +pi branch (documented convention).
    if x_q >= 0:
        ax, ay, z0 = x_q, y_q, 0
    elif y_q >= 0:
        ax, ay, z0 = -x_q, -y_q, PI_Q
    else:
        ax, ay, z0 = -x_q, -y_q, -PI_Q

    # --- Magnitude normalization (left shift).  atan2 is invariant under an
    # isotropic scale, so shifting both folded coordinates up is angle-neutral
    # and lets even a 1-ulp vector drive all 32 rounds (otherwise the floor
    # semantics of sar make -1 >> i == -1 forever and the tail cannot home in
    # on the angle).  k makes the larger component >= 1.0 in Q4.28; the
    # magnitude is divided back by the same 2**k at the end.  When the folded
    # component already reaches 2**31 (a negated INT32_MIN), k is 0 and that
    # value is simply carried in the wide datapath below.
    lead = max(abs(ax), abs(ay))
    norm_shift = max(0, FRAC_BITS + 1 - lead.bit_length())
    ax <<= norm_shift
    ay <<= norm_shift

    # --- 32 vectoring iterations with NO gain pre-scaling: the rotations
    # themselves grow x by the CORDIC gain G_n (~1.647) (pre-scaling by
    # 1/G_n would round 1-ulp inputs to zero).  Both transverse registers are
    # carried in arbitrary-precision integers: after folding, corners have
    # magnitude ~sqrt(2)*2**31 and x ends at G_n times that (~5.0e9),
    # outside the signed 32-bit range; y can transiently reach the same
    # scale.  Clamping either mid-iteration would corrupt the angle.  z is
    # bounded by pi + atan(1) ~ 2.36 rad and stays range-checked.  Gain
    # compensation, normalization undo and saturation happen once, at end.
    x, y, z = ax, ay, z0
    rounds: list[RoundState] = []
    for i in range(ITERATIONS):
        # Vectoring decision: d = -sign(y) (y < 0 -> +1, y >= 0 -> -1), so
        # the same rotation update as sin_cos drives y toward 0.  At y == 0
        # the d = -1 choice only dithers by sub-ulp shifts, as in rotation
        # mode's z == 0 case.  z accumulates the applied rotation: starting
        # from the quadrant offset it converges to atan2(y, x).
        d = -1 if y >= 0 else 1
        x_shift = x >> i
        y_shift = y >> i
        # Same branch formulas as sin_cos; x/y are deliberately wide here.
        if d > 0:
            x_new = x - y_shift
            y_new = y + x_shift
            z_new = check_int32(z - ATAN_TABLE_Q[i], f"round[{i}].z")
        else:
            x_new = x + y_shift
            y_new = y - x_shift
            z_new = check_int32(z + ATAN_TABLE_Q[i], f"round[{i}].z")
        x, y, z = x_new, y_new, z_new
        rounds.append(RoundState(index=i, direction=d, x=x, y=y, z=z))

    # x_final = G_n * 2**k * magnitude.  One ties-to-even Q4.28 multiply
    # removes the gain, a ties-to-even shift undoes the normalization, and
    # the result is clamped to the signed 32-bit maximum.
    gain_comp = mul_q(x, GAIN_INV_Q)
    mag_q, saturated = saturate_int32(rshift_even(gain_comp, norm_shift))
    angle_q = check_int32(z, "output.angle")

    ref_angle, ref_mag = atan2_mag_ref(y_q, x_q)
    with localcontext() as ctx:
        ctx.prec = 50
        angle_err = Decimal(angle_q) / SCALE - ref_angle
        mag_err = Decimal(mag_q) / SCALE - ref_mag

    return Atan2Result(
        angle_q=angle_q,
        mag_q=mag_q,
        saturated=saturated,
        angle_offset_q=z0,
        rounds=tuple(rounds),
        angle_error=+angle_err,
        mag_error=+mag_err,
    )


def atan2(y_q: int, x_q: int) -> int:
    """Thin wrapper over :func:`atan2_mag` returning only the Q4.28 angle."""
    return atan2_mag(y_q, x_q).angle_q


def magnitude(y_q: int, x_q: int) -> int:
    """Thin wrapper over :func:`atan2_mag` returning only the Q4.28 magnitude.

    Saturates at ``INT32_MAX`` when the true magnitude exceeds the format.
    """
    return atan2_mag(y_q, x_q).mag_q


def atan2_mag_radians(y: float, x: float) -> Atan2Result:
    """Convenience wrapper: finite float coordinates -> Q4.28 -> atan2_mag.

    NaN/Infinity raise :class:`NonFiniteValueError`; coordinates outside
    ``[-8.0, 8.0)`` raise :class:`FixedPointOverflowError`.
    """
    return atan2_mag(float_to_q(y, "y"), float_to_q(x, "x"))

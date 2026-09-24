"""32-step CORDIC in signed Q4.28 fixed point (rotation and vectoring modes).

Public interface
----------------
``sin_cos(angle_q) -> CordicResult``
    ``angle_q`` is a signed 32-bit integer holding a Q4.28 radian angle in
    ``[-8.0, 8.0)``.  Returns sin/cos as Q4.28 integers, the per-round state
    trace, and the error against the independent Decimal reference.

``polar(x_q, y_q) -> PolarResult``
    Vectoring mode.  ``x_q``/``y_q`` are signed 32-bit Q4.28 coordinates.
    Returns the magnitude ``hypot(x, y)`` as a Q4.28 integer (saturated to the
    signed 32-bit envelope with ``magnitude_saturated`` set when the true
    value exceeds it), the angle ``atan2(y, x)`` as a Q4.28 integer in
    ``[-pi, pi]``, the quadrant adjustment that was applied, the 32 per-round
    states, and the signed errors against the independent Decimal reference.

Contract highlights (see README for the full contract):

* ``bool`` is rejected as an integer argument; NaN/Infinity are rejected as
  real-valued arguments (see :func:`sin_cos_radians` and :func:`polar_xy`).
* Rotation mode first reduces the angle into ``[-pi/2, pi/2]`` by
  adding/subtracting the quantized pi, recording the quadrant sign flip.
  Vectoring mode first folds the input vector so the CORDIC x-axis input is
  non-negative, recording the quadrant angle offset (0/±pi/2/pi).
* Every right shift is arithmetic, rounding toward negative infinity
  (:func:`cordic_q428.fixedpoint.sar`).
* Every register update is range-checked: any value leaving the signed
  32-bit range raises :class:`FixedPointOverflowError`.  Nothing wraps.  The
  sole documented exception is the *final* magnitude, which by contract
  saturates to ``INT32_MAX``/``INT32_MIN`` instead of raising.
* ``math.sin``/``math.cos``/``math.atan2``/``math.hypot`` are never used
  anywhere in this package.
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
    mul_q,
    require_int,
)
from .reference import polar_ref, sin_cos_ref
from .tables import (
    ATAN_TABLE_Q,
    GAIN_INV_Q,
    GAIN_PREFIX_INV_Q,
    HALF_PI_Q,
    ITERATIONS,
    PI_Q,
)

# The reduction loop below needs at most ceil(8 / pi) = 3 steps for any
# representable Q4.28 angle; the budget is a defensive bound, and exhausting
# it is reported separately from ordinary overflow.
_REDUCTION_BUDGET: int = 8

# Wide envelope for vectoring x/y registers.  The unnormalized worst case is a
# corner whose folded components reach 2**31 (INT32_MIN after the dominant-axis
# fold): final x approaches sqrt(2) * 2**31 * K_32 ~= 5.001e9.  The bound uses
# the slightly larger sqrt(2) * INT32_MAX * 1.65 + slack; the test suite checks
# the observed peak stays inside it.  z always stays within signed int32.
_VECTOR_REG_BOUND: int = int(Decimal(INT32_MAX) * Decimal(2).sqrt() * Decimal("1.65")) + 64

# When the true magnitude is within the Q4.28 envelope, shift-truncation noise
# (at most one integer count per round, amplified by the ~1.647 gain) plus
# gain-table quantization can push the compensated result slightly past
# INT32_MAX.  128 counts is a conservative analytic ceiling (32 rounds * 1.647
# rounded up, plus gain quantization slack); a larger overshoot raises.
_MAG_NOISE_CLIP: int = 128

# Vectoring normalization targets a dominant component with this bit length
# (just under 2**31), leaving headroom for the ~1.647 forward gain so the
# normalized x/y registers never leave the 32-bit wide envelope.  Left-shifting
# is a pure scale change: the angle is invariant and the magnitude shift is
# undone after compensation.
_NORMALIZE_HEAD: int = 31


@dataclass(frozen=True)
class RoundState:
    """State of the CORDIC datapath after one iteration.

    All of ``x``, ``y``, ``z`` are signed Q4.28 integers (in vectoring mode
    ``x``/``y`` may lie in the documented wide envelope).  ``direction`` is
    +1 or -1 for a performed micro-rotation: the sign of ``z`` before the
    update in rotation mode, the sign of ``y`` before the update in vectoring
    mode; it is 0 only in vectoring mode when ``y`` is already aligned
    (no rotation that round).
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


@dataclass(frozen=True)
class PolarResult:
    """Result of :func:`polar` (vectoring-mode CORDIC).

    ``mag_q`` is the Q4.28 integer magnitude ``hypot(x, y)``; it is saturated
    to :data:`INT32_MAX` when the true magnitude exceeds the signed 32-bit
    envelope, in which case ``magnitude_saturated`` is ``True`` (the reported
    ``mag_error`` then reflects the saturation gap honestly).  ``angle_q`` is
    ``atan2(y, x)`` as a Q4.28 integer in ``[-pi, pi]`` (the zero vector
    reports 0).  ``angle_offset_q`` is the quadrant correction added after the
    folded iteration: 0 (x >= 0), ``+PI_Q`` or ``-PI_Q``.  ``rounds`` holds
    the 32 per-iteration states on the folded vector.  ``mag_error`` /
    ``angle_error`` are Decimals: ``output/2**28 - reference`` in real units
    against the independent Decimal oracle in :mod:`cordic_q428.reference`.
    """

    mag_q: int
    angle_q: int
    angle_offset_q: int
    magnitude_saturated: bool
    rounds: tuple[RoundState, ...]
    mag_error: Decimal
    angle_error: Decimal

    @property
    def mag(self) -> float:
        return self.mag_q / SCALE

    @property
    def angle(self) -> float:
        return self.angle_q / SCALE


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


def _validate_coord_q(value: object, name: str) -> int:
    value = require_int(value, name)
    if not (INT32_MIN <= value <= INT32_MAX):
        raise FixedPointOverflowError(
            f"{name}: {value} outside signed 32-bit Q4.28 range "
            f"[{INT32_MIN}, {INT32_MAX}]",
            value=value,
            where=name,
        )
    return value


def _fold_to_x_axis(x_q: int, y_q: int) -> tuple[int, int, int]:
    """Fold the vector onto its dominant axis, returning ``(x', y', off_q)``.

    The fold guarantees ``x' >= |y'| >= 0``-style alignment: the component with
    the larger absolute value lands on the (non-negative) x axis, so the folded
    angle always lies in ``[-pi/4, pi/4]`` — comfortably inside the CORDIC
    convergence sector — and each coordinate axis folds straight onto +x with
    zero micro-rotations (exact magnitude and angle).  The original angle is
    ``folded_angle + off_q``, with ``off_q`` in {0, +-pi/2, +-pi}:

    * ``|x| >= |y|``, ``x >= 0`` -> ``(x,   y)``,  off 0
    * ``|x| >= |y|``, ``x <  0`` -> ``(-x, -y)``,  off +pi (y >= 0) / -pi
    * ``|y| >  |x|``, ``y >  0`` -> ``(y,  -x)``,  off +pi/2
    * ``|y| >  |x|``, ``y <  0`` -> ``(-y,  x)``,  off -pi/2

    The negative x axis (``x < 0, y == 0``) folds with ``off = +pi``; there is
    no signed integer zero, so atan2(0, negative) is defined as +pi (documented
    in the README).  A folded component can reach ``+2**31`` (the ``INT32_MIN``
    case), hence the vector-mode wide register envelope.
    """
    if abs(x_q) >= abs(y_q):
        if x_q >= 0:
            return x_q, y_q, 0
        off = PI_Q if y_q >= 0 else -PI_Q
        return -x_q, -y_q, off
    if y_q > 0:
        return y_q, -x_q, HALF_PI_Q
    return -y_q, x_q, -HALF_PI_Q


def polar(x_q: int, y_q: int) -> PolarResult:
    """Vectoring-mode CORDIC: magnitude and atan2 of a Q4.28 vector.

    ``x_q``/``y_q`` are signed 32-bit Q4.28 integers.  Returns a
    :class:`PolarResult`.  The zero vector ``(0, 0)`` is a defined result:
    magnitude 0 and angle 0.  A true magnitude above the signed 32-bit Q4.28
    envelope (``x**2 + y**2 > INT32_MAX**2``) saturates ``mag_q`` to
    :data:`INT32_MAX` and sets ``magnitude_saturated``; it never raises and
    never wraps.  Every other datapath breach raises
    :class:`FixedPointOverflowError`.
    """
    x_q = _validate_coord_q(x_q, "x_q")
    y_q = _validate_coord_q(y_q, "y_q")

    x, y, offset_q = _fold_to_x_axis(x_q, y_q)

    # Normalization (integer preshifting): a small vector otherwise lives in
    # the low Q4.28 bits and the early ``x >> i`` truncations destroy its
    # angle.  Left-shifting both components by the same amount is a pure scale
    # change — it leaves the angle exactly unchanged and moves the vector near
    # full scale so every micro-rotation keeps full precision.  The shift is
    # undone on the magnitude after gain compensation.  Only left shifts are
    # used (large inputs are left untouched), so the wide register envelope
    # below is never exceeded.
    dominant = max(abs(x), abs(y))
    norm_shift = 0 if dominant == 0 else max(0, _NORMALIZE_HEAD - dominant.bit_length())
    if norm_shift:
        x <<= norm_shift
        y <<= norm_shift

    z = 0
    rounds: list[RoundState] = []
    rotated = 0  # micro-rotations actually performed; always a prefix of 0..32
    for i in range(ITERATIONS):
        # Vectoring decision: drive y toward zero.  y > 0 rotates clockwise
        # (s = +1), y < 0 counter-clockwise (s = -1).  Once y is exactly zero
        # the vector is aligned with the x axis in the integer register set:
        # every remaining round is a no-op (direction 0), x/y/z stay put and
        # no per-round gain factor accrues.  Compensation below uses the
        # inverse gain of exactly the ``rotated`` performed rounds
        # (GAIN_PREFIX_INV_Q), so on-axis inputs and exact angles such as 45
        # degrees land precisely instead of being forced to re-converge.  The
        # zero vector performs zero rotations and stays (0, 0, 0).
        if y > 0:
            s = 1
        elif y < 0:
            s = -1
        else:
            s = 0
        if s == 0:
            x_new, y_new, z_new = x, y, z
        else:
            x_shift = x >> i
            y_shift = y >> i
            if s > 0:
                x_new = x + y_shift
                y_new = y - x_shift
                z_new = z - ATAN_TABLE_Q[i]
            else:
                x_new = x - y_shift
                y_new = y + x_shift
                z_new = z + ATAN_TABLE_Q[i]
            rotated += 1
        if not (-_VECTOR_REG_BOUND <= x_new <= _VECTOR_REG_BOUND):
            raise FixedPointOverflowError(
                f"vectoring round[{i}].x: {x_new} outside the wide vector-mode "
                f"envelope +-{_VECTOR_REG_BOUND}",
                value=x_new,
                where=f"round[{i}].x",
            )
        if not (-_VECTOR_REG_BOUND <= y_new <= _VECTOR_REG_BOUND):
            raise FixedPointOverflowError(
                f"vectoring round[{i}].y: {y_new} outside the wide vector-mode "
                f"envelope +-{_VECTOR_REG_BOUND}",
                value=y_new,
                where=f"round[{i}].y",
            )
        z_new = check_int32(z_new, f"round[{i}].z")
        x, y, z = x_new, y_new, z_new
        rounds.append(RoundState(index=i, direction=s, x=x, y=y, z=z))

    # The folded angle is the negation of the total rotation applied to align
    # the vector with the +x axis; restore the quadrant offset.
    angle_q = check_int32(offset_q - z, "output.angle")

    # Gain compensation for exactly the performed micro-rotations: after
    # ``rotated`` rounds x = K_m * (r * 2**norm_shift), so multiply by 1/K_m
    # (a ties-to-even Q4.28 product) and then undo the normalization shift.
    comp = mul_q(x, GAIN_PREFIX_INV_Q[rotated])
    if norm_shift:
        den = 1 << norm_shift
        mag_scaled, rem = divmod(comp, den)
        if 2 * rem > den or (2 * rem == den and mag_scaled & 1):
            mag_scaled += 1
    else:
        mag_scaled = comp

    # An exact integer predicate decides mathematical saturation so the flag
    # never depends on rounding.  When the true magnitude is within the
    # envelope but shift-truncation noise pushes the computed value a few
    # counts past INT32_MAX (bounded analytically by _MAG_NOISE_CLIP), clip to
    # INT32_MAX without claiming saturation; a larger overshoot is a datapath
    # fault and still raises.
    saturated = x_q * x_q + y_q * y_q > INT32_MAX * INT32_MAX
    if saturated:
        mag_q = INT32_MAX
    elif mag_scaled > INT32_MAX:
        if mag_scaled - INT32_MAX > _MAG_NOISE_CLIP:
            raise FixedPointOverflowError(
                f"output.mag: in-envelope vector compensated to {mag_scaled}, "
                f"{mag_scaled - INT32_MAX} counts over the cap (noise budget "
                f"{_MAG_NOISE_CLIP}) — datapath fault",
                value=mag_scaled,
                where="output.mag",
            )
        mag_q = INT32_MAX
    elif mag_scaled < 0:
        # Folded CORDIC x is non-negative and converges to K_m*r, so the
        # compensated magnitude must be non-negative; anything else is a
        # genuine datapath fault, reported rather than wrapped.
        raise FixedPointOverflowError(
            f"output.mag: negative compensated magnitude {mag_scaled}",
            value=mag_scaled,
            where="output.mag",
        )
    else:
        mag_q = mag_scaled

    ref_angle, ref_mag = polar_ref(x_q, y_q)
    with localcontext() as ctx:
        ctx.prec = 50
        angle_err = Decimal(angle_q) / SCALE - ref_angle
        mag_err = Decimal(mag_q) / SCALE - ref_mag

    return PolarResult(
        mag_q=mag_q,
        angle_q=angle_q,
        angle_offset_q=offset_q,
        magnitude_saturated=saturated,
        rounds=tuple(rounds),
        mag_error=+mag_err,
        angle_error=+angle_err,
    )


def polar_xy(x: float, y: float) -> PolarResult:
    """Convenience wrapper: convert two finite float coordinates to Q4.28.

    NaN and +/-infinity are rejected (:class:`NonFiniteValueError`); values
    outside the Q4.28 signed 32-bit envelope raise
    :class:`FixedPointOverflowError` before any CORDIC work begins.
    """
    return polar(float_to_q(x, "x"), float_to_q(y, "y"))

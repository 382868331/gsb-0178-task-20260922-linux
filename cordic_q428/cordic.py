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
from fractions import Fraction
from numbers import Rational

from .errors import (
    FixedPointOverflowError,
    InvalidArgumentTypeError,
    InvalidArgumentValueError,
    NonFiniteValueError,
    RangeReductionError,
    ToleranceNotGuaranteedError,
)
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

# K_N = prod_{i=0}^{N-1} sqrt(1 + 2**-2i) as an exact Fraction bound.  It is
# used only for the provable round-off analysis of early-terminated runs.
def _build_gain_upper_bound(iterations: int) -> Fraction:
    """Return an exact upper bound on K_n for every n <= iterations.

    ``sqrt(1 + t)`` is bracketed from above by ``1 + t/2`` (the first-order
    binomial term), so ``K_n <= prod_i (1 + 2**(-2i-1))`` exactly.
    """
    bound = Fraction(1)
    for i in range(iterations):
        bound *= Fraction(1) + Fraction(1, 2 ** (2 * i + 1))
    return bound


_K_GLOBAL_BOUND: Fraction = _build_gain_upper_bound(ITERATIONS)

# Tail gain deficit: K_32/K_n - 1 <= sum_{i=n}^{31} 2**(-2i-1) / (the exact
# K factor denominators).  We use the simple exact bound
# K_32/K_n - 1 <= prod_{i=n}^{31}(1 + 2**(-2i-1)) - 1.
_ROUND_TAIL_FACTORS: tuple[Fraction, ...] = tuple(
    Fraction(1) + Fraction(1, 2 ** (2 * i + 1)) for i in range(ITERATIONS)
)


def _gain_deficit_bound(steps: int) -> Fraction:
    """Exact upper bound on K_32/K_steps - 1 (dimensionless Fraction)."""
    product = Fraction(1)
    for i in range(steps, ITERATIONS):
        product *= _ROUND_TAIL_FACTORS[i]
    return product - 1


def _ulp_bound(steps: int, residual_ulp: int) -> Fraction:
    """Provable per-output error bound in ulp after ``steps`` rounds.

    Every bound below is an exact :class:`Fraction`; nothing is rounded.
    The bound counts fixed-point round-off explicitly rather than relying on
    round-to-round output changes.

    Components (all in Q4.28 ulp), per coordinate:

    * angle residual: each quantized table entry a_i differs from the true
      atan(2**-i) by at most 1/2 ulp, so the ideal residual of the *true*
      angle is at most |z_n| + n/2 ulp; the unrotated angle maps directly
      onto a sin/cos coordinate error of the same magnitude;
    * datapath truncation: each round replaces the exact shifted value
      ``v/2**i`` by ``floor(v/2**i)``, an error in (-1, 0] ulp in each of
      the two coordinates (norm <= sqrt(2) ulp).  Errors propagate through
      later rotation rounds with norm amplification bounded by K_32;
    * gain-inverse initialization: x0 is quantized with |error| <= 1/2 ulp
      and the error propagates through n rotations with gain <= K_32;
    * uncompensated gain: an n-step datapath applies gain K_n, but the seed
      carries 1/K_32; the relative amplitude deficit K_32/K_n - 1 acts on
      a unit vector, i.e. contributes at most that many ulp per coordinate;
    * quadrant reduction: every add/subtract of the quantized PI_Q differs
      from true pi by at most 1/2 ulp; reducing any Q4.28 angle needs at
      most 3 such steps, hence at most 3/2 ulp of angle error.
    """
    # Rigorous rational upper bound on sqrt(2): (17/12)**2 = 289/144 > 2.
    sqrt2_upper = Fraction(17, 12)
    angle_residual = Fraction(abs(residual_ulp)) + Fraction(steps, 2)
    truncation = sqrt2_upper * _K_GLOBAL_BOUND * steps
    gain_init = _K_GLOBAL_BOUND / 2
    gain_deficit = _gain_deficit_bound(steps) * SCALE
    reduction = Fraction(3, 2)
    return angle_residual + truncation + gain_init + gain_deficit + reduction


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
    steps: int = ITERATIONS
    tolerance: Fraction | None = None
    error_bound_ulp: Fraction | None = None
    tolerance_met: bool | None = None
    early_stopped: bool = False

    @property
    def sin(self) -> float:
        return self.sin_q / SCALE

    @property
    def cos(self) -> float:
        return self.cos_q / SCALE

    @property
    def error_bound(self) -> Fraction | None:
        """Proven per-coordinate absolute-error bound in real units, or None."""
        if self.error_bound_ulp is None:
            return None
        return self.error_bound_ulp / SCALE


def _coerce_tolerance(tolerance: object, tol_ulp: object) -> Fraction | None:
    """Validate tolerance arguments and return the tolerance as real Fraction.

    ``tolerance`` (real units) accepts int/float/Decimal/Fraction; ``tol_ulp``
    accepts a positive int counted in Q4.28 ulp.  At most one may be given.
    bool, NaN/Infinity and non-positive values are rejected, not clamped.
    """
    if tolerance is not None and tol_ulp is not None:
        raise InvalidArgumentValueError(
            "tolerance and tol_ulp are mutually exclusive; pass at most one"
        )
    if tolerance is not None:
        if isinstance(tolerance, bool):
            raise InvalidArgumentTypeError("tolerance: bool is not a valid tolerance")
        if isinstance(tolerance, Rational):
            value = Fraction(tolerance)
        elif isinstance(tolerance, (float, Decimal)):
            t = float(tolerance)
            if t != t or t in (float("inf"), float("-inf")):
                raise NonFiniteValueError(
                    f"tolerance: must be finite and positive, got {tolerance!r}"
                )
            value = Fraction(t)
        else:
            raise InvalidArgumentTypeError(
                f"tolerance: expected int, float or Decimal, got {type(tolerance).__name__}"
            )
        if value <= 0:
            raise InvalidArgumentValueError(
                f"tolerance: must be strictly positive, got {tolerance!r}"
            )
        return value
    if tol_ulp is not None:
        if isinstance(tol_ulp, bool) or not isinstance(tol_ulp, int):
            raise InvalidArgumentTypeError(
                f"tol_ulp: expected a positive int number of Q4.28 ulp, got {tol_ulp!r}"
            )
        if tol_ulp <= 0:
            raise InvalidArgumentValueError(
                f"tol_ulp: must be strictly positive, got {tol_ulp}"
            )
        return Fraction(tol_ulp, SCALE)
    return None


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


def sin_cos(
    angle_q: int,
    *,
    tolerance: int | float | Decimal | None = None,
    tol_ulp: int | None = None,
) -> CordicResult:
    """Compute sin and cos of a Q4.28 angle with rotation-mode CORDIC.

    By default all ``ITERATIONS`` (32) rounds execute, preserving the original
    full computation.  When ``tolerance`` (absolute error in real units) or
    ``tol_ulp`` (absolute error counted in Q4.28 ulp) is given, the routine
    may stop as soon as a *provable* per-coordinate error bound - computed
    from the residual angle and accumulated fixed-point round-off, never
    merely from round-to-round changes - is within the request.  The result
    reports ``steps``, ``error_bound_ulp`` and ``tolerance_met``.

    If even the full 32-step bound cannot satisfy the request, the routine
    raises :class:`ToleranceNotGuaranteedError` (attaching the full result)
    rather than relaxing the requirement.
    """
    angle_q = _validate_angle_q(angle_q)
    tol = _coerce_tolerance(tolerance, tol_ulp)
    reduced, sign_flip = _reduce_to_half_pi(angle_q)

    x = GAIN_INV_Q
    y = 0
    z = reduced
    rounds: list[RoundState] = []
    steps = ITERATIONS
    early_stopped = False
    best_bound_ulp: Fraction | None = None
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
        bound_ulp = _ulp_bound(i + 1, z)
        if best_bound_ulp is None or bound_ulp < best_bound_ulp:
            best_bound_ulp = bound_ulp
        if tol is not None and bound_ulp / SCALE <= tol:
            steps = i + 1
            early_stopped = True
            break

    bound_ulp = _ulp_bound(steps, rounds[steps - 1].z)
    # The bound is non-monotonic in the step count (residual z can overshoot),
    # so the best guarantee that the representation could provide is the
    # minimum over every reached round, not necessarily the last one.
    guaranteed_ulp = bound_ulp if best_bound_ulp is None else min(bound_ulp, best_bound_ulp)

    cos_q = check_int32(sign_flip * x, "output.cos")
    sin_q = check_int32(sign_flip * y, "output.sin")

    ref_sin, ref_cos = sin_cos_ref(angle_q)
    with localcontext() as ctx:
        ctx.prec = 50
        sin_err = Decimal(sin_q) / SCALE - ref_sin
        cos_err = Decimal(cos_q) / SCALE - ref_cos

    result = CordicResult(
        sin_q=sin_q,
        cos_q=cos_q,
        reduced_q=reduced,
        sign_flip=sign_flip,
        rounds=tuple(rounds),
        sin_error=+sin_err,
        cos_error=+cos_err,
        steps=steps,
        tolerance=tol,
        error_bound_ulp=bound_ulp,
        tolerance_met=None if tol is None else guaranteed_ulp / SCALE <= tol,
        early_stopped=early_stopped,
    )
    if tol is not None and not result.tolerance_met:
        raise ToleranceNotGuaranteedError(
            f"requested absolute tolerance {tol} cannot be guaranteed: the "
            f"tightest provable bound over all {ITERATIONS} rounds is "
            f"{guaranteed_ulp / SCALE} ({float(guaranteed_ulp):.3f} ulp)",
            steps=ITERATIONS,
            bound=guaranteed_ulp / SCALE,
            tolerance=tol,
            result=result,
        )
    return result


def sin_cos_radians(
    angle: float,
    *,
    tolerance: int | float | Decimal | None = None,
    tol_ulp: int | None = None,
) -> CordicResult:
    """Convenience wrapper: convert a finite float radian angle to Q4.28 first.

    NaN and +/-infinity are rejected (:class:`NonFiniteValueError`); values
    outside ``[-8.0, 8.0)`` raise :class:`FixedPointOverflowError` rather than
    being silently wrapped.
    """
    return sin_cos(float_to_q(angle, "angle"), tolerance=tolerance, tol_ulp=tol_ulp)

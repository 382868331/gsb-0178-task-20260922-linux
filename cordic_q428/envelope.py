"""Conservative error envelopes for the Q4.28 rotation CORDIC.

This module adds *proof information* on top of :mod:`cordic_q428.cordic`.
Instead of only reporting the a-posteriori difference against the Decimal
reference, it propagates lower/upper bounds through every round of the
fixed-point iteration and returns an **integer envelope** that is guaranteed
to contain both

* the true ``sin``/``cos`` of the *quantized* input angle
  ``angle_q / 2**28`` — the envelope is tied to the explicit quantized input
  value; no unknown pre-quantization angle error is ever mixed in, and
* the integer outputs actually produced by the CORDIC core,

so ``|output - true| <= hi - lo`` holds by construction.

Error sources, accounted for separately (each can be toggled in the engine,
and the headline numbers report each source in isolation):

* **gain quantization** — ``GAIN_INV_Q`` vs the true 32-step inverse gain
  (<= 0.5 ulp; the exact error is recomputed from the 80-digit Decimal
  :data:`cordic_q428.tables.GAIN_INV_DEC`);
* **angle-table quantization** — each ``ATAN_TABLE_Q[i]`` vs
  ``atan(2**-i) * 2**28`` (<= 0.5 ulp per entry; exact per-entry errors from
  :data:`cordic_q428.tables.ATAN_TABLE_DEC`);
* **arithmetic-shift truncation** — ``sar(v, i) == floor(v / 2**i)`` loses
  ``f`` ulp with ``0 <= f <= 1 - 2**-i`` on every shifted register, every
  round.

The residual angle left after 32 rounds (iteration termination, not one of
the three sources above) is bounded separately and applied as a final
exact-interval rotation; it is reported as its own component
(``residual_rotation_ulp``) instead of being silently folded into the
sources.

Branching is handled conservatively at both sign decisions of the datapath:

* **quadrant reduction is piecewise**: for an angle landing exactly on
  ``+/-HALF_PI_Q`` both legal reduction paths (stop there, or take one more
  ``-/+ pi`` step with the quadrant sign flipped) are evaluated and their
  envelopes unioned;
* **inside the rotation**, the direction ``d`` is chosen from the sign of the
  interval of the true remaining angle ``z``: if that interval straddles 0,
  both ``d = +1`` and ``d = -1`` are propagated and unioned
  (``SinCosEnvelope.union_rounds`` lists where this happened).

All bounds are exact :class:`fractions.Fraction` values in ulp (2**-28)
units.  Nothing is saturated or clamped to 32 bits: a genuinely weak proof
yields a genuinely wide envelope, while overflow of the actual datapath is
still reported by the core as :class:`FixedPointOverflowError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .fixedpoint import SCALE
from .tables import (
    ATAN_TABLE_DEC,
    ATAN_TABLE_Q,
    GAIN_INV_DEC,
    GAIN_INV_Q,
    HALF_PI_Q,
    ITERATIONS,
    PI_Q,
)

# Safety margin covering the 80-digit Decimal truncation of the tables when
# their quantization errors are restated as exact Fractions.
_DEC_EPS = Fraction(1, 10**30)

_ZERO = Fraction(0)


def _table_errors_ulp() -> tuple[Fraction, ...]:
    """Exact |atan(2**-i)*2**28 - ATAN_TABLE_Q[i]| per entry, plus epsilon."""
    return tuple(
        abs(Fraction(dec * SCALE) - q) + _DEC_EPS
        for dec, q in zip(ATAN_TABLE_DEC, ATAN_TABLE_Q)
    )


#: Per-entry angle-table quantization error in ulp (each <= 0.5 + 1e-30).
TABLE_ERR_ULP: tuple[Fraction, ...] = _table_errors_ulp()

#: Exact gain-inverse quantization error in ulp (<= 0.5 + 1e-30).
GAIN_ERR_ULP: Fraction = abs(Fraction(GAIN_INV_DEC * SCALE) - GAIN_INV_Q) + _DEC_EPS


@dataclass(frozen=True)
class IntervalBounds:
    """Exact closed interval ``[lo, hi]`` in ulp (2**-28) units."""

    lo: Fraction
    hi: Fraction

    @property
    def width(self) -> Fraction:
        return self.hi - self.lo

    def __contains__(self, value: object) -> bool:
        return self.lo <= value <= self.hi


def _union(a: IntervalBounds, b: IntervalBounds) -> IntervalBounds:
    return IntervalBounds(min(a.lo, b.lo), max(a.hi, b.hi))


def _mul(a: IntervalBounds, b: IntervalBounds) -> IntervalBounds:
    products = (a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi)
    return IntervalBounds(min(products), max(products))


def _floor_frac(v: Fraction) -> int:
    return v.numerator // v.denominator


def _ceil_frac(v: Fraction) -> int:
    return -((-v.numerator) // v.denominator)


@dataclass(frozen=True)
class RoundEnvelope:
    """Per-round bound trace of the envelope engine.

    ``directions`` is ``(1,)`` or ``(-1,)`` when the sign of the true
    remaining-angle interval decided the branch, or ``(1, -1)`` when the
    interval straddled 0 and both legal rotations were unioned.  ``x``/``y``
    intervals contain both the ideal real trajectory and the actual integer
    registers after this round; ``z`` contains both the true remaining angle
    and the integer ``z`` register.
    """

    index: int
    directions: tuple[int, ...]
    x: IntervalBounds
    y: IntervalBounds
    z: IntervalBounds


@dataclass(frozen=True)
class EnvelopeSources:
    """Per-source error accounting, in ulp (2**-28) units, as Fractions.

    ``gain_quantization_ulp`` / ``table_quantization_ulp`` /
    ``shift_truncation_ulp`` are the envelope widths (max of sin/cos)
    produced by the propagation engine with *only* that error source enabled
    and the others idealized — a conservative per-source attribution.
    ``residual_rotation_ulp`` bounds how far the final rotation by the
    bounded residual angle can move the endpoint (iteration termination).
    """

    gain_quantization_ulp: Fraction
    table_quantization_ulp: Fraction
    shift_truncation_ulp: Fraction
    residual_rotation_ulp: Fraction


@dataclass(frozen=True)
class SinCosEnvelope:
    """Proven integer envelope for one quantized input angle.

    ``sin_lo <= sin(angle_q/2**28) * 2**28 <= sin_hi`` and likewise for cos,
    and the core's integer outputs satisfy the same bounds.  ``rounds`` is
    the per-round bound trace of the all-sources run (primary reduction
    path); ``union_rounds`` lists rounds where the direction branch was
    unioned; ``boundary_union`` records that the input sat exactly on a
    quadrant-reduction boundary and both legal reduction paths were unioned.
    """

    sin_lo: int
    sin_hi: int
    cos_lo: int
    cos_hi: int
    reduced_q: int
    sign_flip: int
    boundary_union: bool
    union_rounds: tuple[int, ...]
    z_residual: IntervalBounds
    sources: EnvelopeSources
    rounds: tuple[RoundEnvelope, ...]

    @property
    def sin_width(self) -> int:
        return self.sin_hi - self.sin_lo

    @property
    def cos_width(self) -> int:
        return self.cos_hi - self.cos_lo

    def contains(self, sin_q: int, cos_q: int) -> bool:
        """True iff the integer outputs lie inside the envelope."""
        return self.sin_lo <= sin_q <= self.sin_hi and self.cos_lo <= cos_q <= self.cos_hi


def _shift_interval(v: IntervalBounds, i: int, shift_on: bool) -> IntervalBounds:
    """Bound both ``value * 2**-i`` (ideal) and ``sar(value, i)`` (register).

    ``sar(v, i) = floor(v / 2**i)`` loses ``f`` ulp, ``0 <= f <= 1 - 2**-i``,
    so the register shift lies in ``[v*2**-i - (1 - 2**-i), v*2**-i]``.
    """
    f = Fraction(1, 1 << i)
    lo = v.lo * f
    hi = v.hi * f
    if shift_on:
        lo -= 1 - f
    return IntervalBounds(lo, hi)


def _step(
    x: IntervalBounds,
    y: IntervalBounds,
    z: IntervalBounds,
    i: int,
    d: int,
    table_on: bool,
    shift_on: bool,
) -> tuple[IntervalBounds, IntervalBounds, IntervalBounds]:
    sx = _shift_interval(x, i, shift_on)
    sy = _shift_interval(y, i, shift_on)
    if d > 0:
        x_new = IntervalBounds(x.lo - sy.hi, x.hi - sy.lo)
        y_new = IntervalBounds(y.lo + sx.lo, y.hi + sx.hi)
    else:
        x_new = IntervalBounds(x.lo + sy.lo, x.hi + sy.hi)
        y_new = IntervalBounds(y.lo - sx.hi, y.hi - sx.lo)
    tq = TABLE_ERR_ULP[i] if table_on else _ZERO
    entry = Fraction(ATAN_TABLE_Q[i])
    if d > 0:
        z_new = IntervalBounds(z.lo - entry - tq, z.hi - entry + tq)
    else:
        z_new = IntervalBounds(z.lo + entry - tq, z.hi + entry + tq)
    return x_new, y_new, z_new


def _propagate(
    reduced_q: int,
    gain_on: bool,
    table_on: bool,
    shift_on: bool,
) -> tuple[IntervalBounds, IntervalBounds, IntervalBounds, tuple[RoundEnvelope, ...], tuple[int, ...]]:
    """Propagate intervals through the 32 rounds; union at branch boundaries."""
    g = GAIN_ERR_ULP if gain_on else _ZERO
    x = IntervalBounds(Fraction(GAIN_INV_Q) - g, Fraction(GAIN_INV_Q) + g)
    y = IntervalBounds(_ZERO, _ZERO)
    z = IntervalBounds(Fraction(reduced_q), Fraction(reduced_q))
    rounds: list[RoundEnvelope] = []
    union_rounds: list[int] = []
    for i in range(ITERATIONS):
        if z.lo >= 0:
            dirs = (1,)
        elif z.hi < 0:
            dirs = (-1,)
        else:
            # True remaining angle may have either sign: both rotations are
            # legal outcomes; propagate both and take the union envelope.
            dirs = (1, -1)
            union_rounds.append(i)
        variants = [_step(x, y, z, i, d, table_on, shift_on) for d in dirs]
        x = variants[0][0]
        y = variants[0][1]
        z = variants[0][2]
        for vx, vy, vz in variants[1:]:
            x = _union(x, vx)
            y = _union(y, vy)
            z = _union(z, vz)
        rounds.append(RoundEnvelope(index=i, directions=dirs, x=x, y=y, z=z))
    return x, y, z, tuple(rounds), tuple(union_rounds)


def _close_envelope(
    x: IntervalBounds,
    y: IntervalBounds,
    z: IntervalBounds,
    sign_flip: int,
) -> tuple[IntervalBounds, IntervalBounds, Fraction]:
    """Rotate the endpoint bounds by the bounded residual angle ``z``.

    After 32 rounds the ideal endpoint equals ``(cos(theta), sin(theta))``
    with ``theta = reduced - z_residual``; the true value is therefore the
    endpoint rotated by ``z_residual``.  The residual interval is widened to
    include 0 so the envelope also contains the un-rotated register output.
    Uses ``|sin z - z| <= |z|**3/6`` and ``1 - z**2/2 <= cos z <= 1``.
    """
    b = max(abs(z.lo), abs(z.hi))  # ulp
    z_lo = min(z.lo, _ZERO)
    z_hi = max(z.hi, _ZERO)
    scale = Fraction(SCALE)
    sin_corr = b**3 / (6 * SCALE * SCALE)
    sz = IntervalBounds(z_lo - sin_corr, z_hi + sin_corr)  # sin(z) * SCALE, ulp
    cz = IntervalBounds(scale - b * b / (2 * SCALE), scale)  # cos(z) * SCALE, ulp
    x_cz = _mul(x, cz)
    y_sz = _mul(y, sz)
    x_sz = _mul(x, sz)
    y_cz = _mul(y, cz)
    cos_i = IntervalBounds((x_cz.lo - y_sz.hi) / scale, (x_cz.hi - y_sz.lo) / scale)
    sin_i = IntervalBounds((x_sz.lo + y_cz.lo) / scale, (x_sz.hi + y_cz.hi) / scale)
    if sign_flip < 0:
        cos_i = IntervalBounds(-cos_i.hi, -cos_i.lo)
        sin_i = IntervalBounds(-sin_i.hi, -sin_i.lo)
    # Bound on how far the residual rotation can move the endpoint.
    norm = max(abs(x.lo), abs(x.hi), abs(y.lo), abs(y.hi))
    rot = norm * b / SCALE
    return sin_i, cos_i, rot


@dataclass(frozen=True)
class _Run:
    sin_i: IntervalBounds
    cos_i: IntervalBounds
    rounds: tuple[RoundEnvelope, ...]
    union_rounds: tuple[int, ...]
    z_residual: IntervalBounds
    rot: Fraction

    def union(self, other: "_Run") -> "_Run":
        return _Run(
            sin_i=_union(self.sin_i, other.sin_i),
            cos_i=_union(self.cos_i, other.cos_i),
            rounds=self.rounds,  # primary path keeps the representative trace
            union_rounds=tuple(sorted(set(self.union_rounds) | set(other.union_rounds))),
            z_residual=_union(self.z_residual, other.z_residual),
            rot=max(self.rot, other.rot),
        )


def _engine_run(
    reduced_q: int,
    sign_flip: int,
    gain_on: bool,
    table_on: bool,
    shift_on: bool,
) -> _Run:
    x, y, z, rounds, union_rounds = _propagate(reduced_q, gain_on, table_on, shift_on)
    sin_i, cos_i, rot = _close_envelope(x, y, z, sign_flip)
    return _Run(sin_i, cos_i, rounds, union_rounds, z, rot)


def _reduction_paths(reduced_q: int, sign_flip: int) -> tuple[tuple[int, int], ...]:
    """Legal reduction outcomes for a quantized angle already in range.

    Exactly on ``+/-HALF_PI_Q`` the reduction loop stops, but taking one more
    ``-/+ pi`` step (flipping the quadrant sign) is an equally legal
    piecewise representation of the same angle; both are returned so the
    caller can union the two paths' envelopes.
    """
    paths = [(reduced_q, sign_flip)]
    if reduced_q == HALF_PI_Q:
        paths.append((reduced_q - PI_Q, -sign_flip))
    elif reduced_q == -HALF_PI_Q:
        paths.append((reduced_q + PI_Q, -sign_flip))
    return tuple(paths)


def envelope_for_reduced(reduced_q: int, sign_flip: int) -> SinCosEnvelope:
    """Compute the proven sin/cos envelope for a reduced Q4.28 angle.

    ``reduced_q`` must already lie in ``[-HALF_PI_Q, HALF_PI_Q]`` and
    ``sign_flip`` must be +1 or -1 (the quadrant sign factored out by the
    reduction).  This is a pure function: exact Fraction arithmetic only.
    """
    paths = _reduction_paths(reduced_q, sign_flip)
    boundary = len(paths) > 1

    def run(gain_on: bool, table_on: bool, shift_on: bool) -> _Run:
        combined: _Run | None = None
        for rq, sf in paths:
            r = _engine_run(rq, sf, gain_on, table_on, shift_on)
            combined = r if combined is None else combined.union(r)
        assert combined is not None
        return combined

    full = run(True, True, True)
    gain = run(True, False, False)
    table = run(False, True, False)
    shift = run(False, False, True)

    sources = EnvelopeSources(
        gain_quantization_ulp=max(gain.sin_i.width, gain.cos_i.width),
        table_quantization_ulp=max(table.sin_i.width, table.cos_i.width),
        shift_truncation_ulp=max(shift.sin_i.width, shift.cos_i.width),
        residual_rotation_ulp=full.rot,
    )
    return SinCosEnvelope(
        sin_lo=_floor_frac(full.sin_i.lo),
        sin_hi=_ceil_frac(full.sin_i.hi),
        cos_lo=_floor_frac(full.cos_i.lo),
        cos_hi=_ceil_frac(full.cos_i.hi),
        reduced_q=reduced_q,
        sign_flip=sign_flip,
        boundary_union=boundary,
        union_rounds=full.union_rounds,
        z_residual=full.z_residual,
        sources=sources,
        rounds=full.rounds,
    )

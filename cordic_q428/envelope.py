"""Conservative error-envelope propagation for the Q4.28 CORDIC core.

This module is *proof-only*: it never changes the rotation algorithm in
:mod:`cordic_q428.cordic`.  It re-runs the same recurrence in exact rational
interval arithmetic (:class:`fractions.Fraction`) and propagates lower/upper
bounds through every round, so that the final interval provably contains both

* the actual fixed-point register trajectory of :func:`sin_cos`, and
* the mathematical ``sin``/``cos`` of the *explicit quantized input angle*
  ``angle_q / 2**28`` (no pre-quantization angle error is mixed in — the
  envelope is a statement about the integer the caller passed, nothing else).

Three error sources are modelled, each as an exact rational interval:

* **gain quantization** — the true inverse CORDIC gain lies within half a ulp
  of ``GAIN_INV_Q`` (ties-to-even quantization, plus a 1e-60 ulp margin that
  absorbs the 80-digit Decimal evaluation error of the table generator), so
  the initial ``x`` interval is ``GAIN_INV_Q +/- (1/2 + 1e-60)`` ulp.
* **angle-table quantization** — every ``atan(2**-i)`` table entry and the
  quantized ``pi`` used by quadrant reduction carry the same half-ulp
  quantization interval.  Each ``z`` update subtracts the interval
  ``ATAN_TABLE_Q[i] +/- (1/2 + 1e-60)`` ulp, and the reduced angle is widened
  by ``k * (1/2 + 1e-60)`` ulp where ``k`` is the number of pi subtractions
  the reduction performed (the true angle differs from the reduced one by
  ``k`` times the pi quantization error).
* **arithmetic right-shift truncation** — ``sar`` rounds toward negative
  infinity, losing ``v/2**i - floor(v/2**i) in [0, 1 - 2**-i)`` ulp per shift.
  Each shift is propagated as the union of the floored shift (the hardware
  behaviour, applied to the integer register grid inside the interval) and
  the exact division (the ideal behaviour).  The per-round truncation gap is
  recorded in :class:`RoundEnvelope` so tests can verify it never exceeds
  ``1 - 2**-i``.

Branch boundaries (quadrant reduction and the per-round direction decision
``d = 1 if z >= 0 else -1``) are handled by *union*: whenever an interval
straddles a branch condition, both legal paths are propagated and their
intervals merged, so the envelope stays valid no matter which side the exact
trajectory falls on.

After the last round the residual angle ``z`` (the iteration stops with
``|z| <= atan(2**-31)`` plus accumulated table-quantization width) is folded
back into the ``x``/``y`` intervals: ``sin``/``cos`` are 1-Lipschitz, so the
true value at the reduced angle lies within ``max(|z_lo|, |z_hi|)`` of the
final vector.

Overflow is *not* hidden by saturation: if any propagated endpoint leaves the
signed 32-bit range, :class:`FixedPointOverflowError` is raised (a
conservative proof failure), exactly matching the core's no-wrap contract.

To attribute the envelope to its causes, :func:`compute_envelope` runs the
propagation four times: once with all sources enabled (the reported envelope)
and once per isolated source.  The isolated widths are reported in
:class:`ErrorSources`; they need not sum to the total (each isolated run
carries its own final residual-angle term), but each is a valid standalone
bound for that source alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .errors import FixedPointOverflowError, RangeReductionError
from .fixedpoint import INT32_MAX, INT32_MIN, require_int
from .tables import ATAN_TABLE_Q, GAIN_INV_Q, HALF_PI_Q, ITERATIONS, PI_Q

# Half-width of the quantization interval of any ties-to-even quantized Q4.28
# constant: 1/2 ulp, plus a 1e-60 ulp margin absorbing the 80-digit Decimal
# evaluation error of the table generator (itself far below 1e-60 ulp).
QUANT_HALF_ULP: Fraction = Fraction(1, 2) + Fraction(1) / 10**60

# Defensive budget for the interval reduction, mirroring the core.
_REDUCTION_BUDGET: int = 8

# Defensive cap on simultaneously live paths.  Branch straddles split paths;
# in practice a point angle stays under a few dozen, but if the cap is ever
# hit, same-sign paths are merged into their box union (conservative, still
# correct — never silently dropped).
_PATH_BUDGET: int = 4096

_INT32_LO = Fraction(INT32_MIN)
_INT32_HI = Fraction(INT32_MAX)
_HALF_PI = Fraction(HALF_PI_Q)
_PI = Fraction(PI_Q)


@dataclass(frozen=True)
class RoundEnvelope:
    """Proof record for one CORDIC iteration of the full (all-sources) run.

    ``directions`` lists the direction branches that were live *before* this
    round's update: ``(1,)`` or ``(-1,)`` when the z interval stayed on one
    side of the branch boundary, ``(1, -1)`` when it straddled zero and both
    legal paths were unioned.  ``x_width``/``y_width``/``z_width`` are the
    total envelope widths after this round, in ulp.  ``shift_trunc`` is the
    largest shift-truncation gap ``v/2**i - floor(v/2**i)`` injected by this
    round's two shifts; it must satisfy ``0 <= shift_trunc <= 1 - 2**-i``.
    """

    index: int
    directions: tuple[int, ...]
    x_width: Fraction
    y_width: Fraction
    z_width: Fraction
    shift_trunc: Fraction


@dataclass(frozen=True)
class SourceWidth:
    """Isolated contribution of one error source, in ulp (exact rationals)."""

    sin: Fraction
    cos: Fraction


@dataclass(frozen=True)
class ErrorSources:
    """Per-source envelope half-widths (ulp) from the isolated propagation runs."""

    gain_quantization: SourceWidth
    table_quantization: SourceWidth
    shift_truncation: SourceWidth


@dataclass(frozen=True)
class EnvelopeResult:
    """Conservative integer envelope for sin/cos of one quantized angle.

    ``sin_lo``/``sin_hi``/``cos_lo``/``cos_hi`` are signed Q4.28 integers:
    the exact rational bounds (``*_exact``) rounded outward.  Both the
    fixed-point outputs of :func:`sin_cos` and the true
    ``sin``/``cos`` of ``angle_q / 2**28`` (times ``2**28``) lie inside.
    ``sources`` attributes the width to gain quantization, angle-table
    quantization and shift truncation; ``rounds`` holds the 32 per-round
    proof records of the all-sources run.
    """

    angle_q: int
    reduced_q: int
    sign_flip: int
    sin_lo: int
    sin_hi: int
    cos_lo: int
    cos_hi: int
    sin_lo_exact: Fraction
    sin_hi_exact: Fraction
    cos_lo_exact: Fraction
    cos_hi_exact: Fraction
    sources: ErrorSources
    rounds: tuple[RoundEnvelope, ...]

    @property
    def sin_width_ulp(self) -> Fraction:
        """Total sin envelope width in ulp (exact)."""
        return self.sin_hi_exact - self.sin_lo_exact

    @property
    def cos_width_ulp(self) -> Fraction:
        """Total cos envelope width in ulp (exact)."""
        return self.cos_hi_exact - self.cos_lo_exact

    def contains(self, sin_value: object, cos_value: object) -> bool:
        """Check ``sin_value``/``cos_value`` (Q4.28 scale) against the envelope.

        Accepts ints (fixed-point outputs) or Decimal/Fraction reals already
        scaled by ``2**28`` (e.g. ``reference * SCALE``); comparison uses the
        exact rational bounds.
        """
        s = Fraction(sin_value)  # type: ignore[arg-type]
        c = Fraction(cos_value)  # type: ignore[arg-type]
        return (
            self.sin_lo_exact <= s <= self.sin_hi_exact
            and self.cos_lo_exact <= c <= self.cos_hi_exact
        )


class _Path:
    """One live propagation path: a quadrant sign plus x/y/z intervals (ulp)."""

    __slots__ = ("sign", "steps", "xlo", "xhi", "ylo", "yhi", "zlo", "zhi")

    def __init__(self, sign: int, steps: int, xlo: Fraction, xhi: Fraction,
                 ylo: Fraction, yhi: Fraction, zlo: Fraction, zhi: Fraction):
        self.sign = sign
        self.steps = steps  # number of pi subtractions applied in reduction
        self.xlo, self.xhi = xlo, xhi
        self.ylo, self.yhi = ylo, yhi
        self.zlo, self.zhi = zlo, zhi


def _floor_shift(v: Fraction, i: int) -> Fraction:
    """Exact ``floor(v / 2**i)`` — the sar result for the interval endpoint."""
    return Fraction(v.numerator // (v.denominator << i))


def _shift_union(lo: Fraction, hi: Fraction, i: int, truncate: bool) -> tuple[Fraction, Fraction, Fraction]:
    """Propagate one arithmetic right shift; return (lo, hi, trunc_gap).

    With ``truncate`` off, the shift is the ideal exact division.  With it on,
    the result is the union of the floored shift (hardware) and the exact
    division (ideal).  The hardware register is an integer, so the smallest
    register value inside the interval is ``ceil(lo)`` and its floored shift
    is ``floor(ceil(lo)/2**i)``; the lower bound is the minimum of that and
    the ideal ``lo/2**i``.  ``trunc_gap`` is the width the flooring adds
    beyond exact division, provably in ``[0, 1 - 2**-i]``.
    """
    denom = 1 << i
    elo, ehi = lo / denom, hi / denom
    if not truncate:
        return elo, ehi, Fraction(0)
    ceil_lo = Fraction(-((-lo.numerator) // lo.denominator))
    flo = _floor_shift(ceil_lo, i)
    lower = min(elo, flo)
    return lower, ehi, elo - lower


def _check_range(path: _Path, where: str) -> None:
    """Raise on overflow — the envelope never saturates to hide one."""
    for name, v in (("xlo", path.xlo), ("xhi", path.xhi),
                    ("ylo", path.ylo), ("yhi", path.yhi),
                    ("zlo", path.zlo), ("zhi", path.zhi)):
        if v < _INT32_LO or v > _INT32_HI:
            raise FixedPointOverflowError(
                f"envelope endpoint left signed 32-bit range: {name}={float(v)!r}",
                value=float(v),
                where=where,
            )


def _reduce_interval(zlo: Fraction, zhi: Fraction) -> list[_Path]:
    """Reduce an angle interval to [-pi/2, pi/2], segment-wise.

    Returns the live paths ``(sign, steps, z-interval)``.  A point interval
    (the normal case — the input is one explicit quantized angle) reduces
    deterministically exactly like the core.  An interval straddling a
    reduction boundary is split so both legal outcomes (reduced / not
    reduced) survive in the union.
    """
    half, pi = _HALF_PI, _PI
    pending = [_Path(1, 0, Fraction(0), Fraction(0), Fraction(0), Fraction(0), zlo, zhi)]
    done: list[_Path] = []
    budget = _REDUCTION_BUDGET
    while pending:
        p = pending.pop()
        if p.zlo >= -half and p.zhi <= half:
            done.append(p)
            continue
        if budget <= 0:
            raise RangeReductionError(
                f"envelope quadrant reduction did not converge within "
                f"{_REDUCTION_BUDGET} steps"
            )
        if p.zlo > half:
            # Entire interval above the boundary: subtract pi, flip sign.
            p.zlo -= pi
            p.zhi -= pi
            p.sign = -p.sign
            p.steps += 1
            budget -= 1
            pending.append(p)
        elif p.zhi < -half:
            p.zlo += pi
            p.zhi += pi
            p.sign = -p.sign
            p.steps += 1
            budget -= 1
            pending.append(p)
        elif p.zhi > half:
            # Straddles the upper boundary: keep the inside part, reduce the
            # rest.  The inside part re-enters the queue because a very wide
            # interval could straddle the lower boundary too.
            inside = _Path(p.sign, p.steps, p.xlo, p.xhi, p.ylo, p.yhi, p.zlo, half)
            outside = _Path(-p.sign, p.steps + 1, p.xlo, p.xhi, p.ylo, p.yhi,
                            half - pi, p.zhi - pi)
            budget -= 1
            pending.append(inside)
            pending.append(outside)
        else:
            # Straddles the lower boundary.
            inside = _Path(p.sign, p.steps, p.xlo, p.xhi, p.ylo, p.yhi, -half, p.zhi)
            outside = _Path(-p.sign, p.steps + 1, p.xlo, p.xhi, p.ylo, p.yhi,
                            p.zlo + pi, -half + pi)
            budget -= 1
            pending.append(inside)
            pending.append(outside)
    return done


def _propagate(angle_q: int, *, gain_err: bool, table_err: bool,
               shift_err: bool, record: bool) -> tuple[list[_Path], tuple[RoundEnvelope, ...]]:
    """Run the 32-round interval recurrence; return final paths and records."""
    # Initial state: x carries the gain-quantization interval (or the exact
    # register value when that source is isolated off); y = 0, z = angle.
    if gain_err:
        xlo = Fraction(GAIN_INV_Q) - QUANT_HALF_ULP
        xhi = Fraction(GAIN_INV_Q) + QUANT_HALF_ULP
    else:
        xlo = xhi = Fraction(GAIN_INV_Q)
    zero = Fraction(0)

    paths = _reduce_interval(Fraction(angle_q), Fraction(angle_q))
    for p in paths:
        p.xlo, p.xhi = xlo, xhi
        p.ylo = p.yhi = zero
        # Angle-table quantization of the pi used in reduction: the true
        # angle differs from the reduced register value by steps * (pi_q -
        # pi_true), |pi_q - pi_true| <= QUANT_HALF_ULP ulp.
        if table_err and p.steps:
            widen = p.steps * QUANT_HALF_ULP
            p.zlo -= widen
            p.zhi += widen

    rounds: list[RoundEnvelope] = []
    for i in range(ITERATIONS):
        atan_lo = atan_hi = Fraction(ATAN_TABLE_Q[i])
        if table_err:
            atan_lo -= QUANT_HALF_ULP
            atan_hi += QUANT_HALF_ULP
        next_paths: list[_Path] = []
        round_trunc = Fraction(0)
        round_dirs: set[int] = set()
        for p in paths:
            dirs: list[int] = []
            if p.zhi >= 0:
                dirs.append(1)
            if p.zlo < 0:
                dirs.append(-1)
            round_dirs.update(dirs)
            for d in dirs:
                xs_lo, xs_hi, tx = _shift_union(p.xlo, p.xhi, i, shift_err)
                ys_lo, ys_hi, ty = _shift_union(p.ylo, p.yhi, i, shift_err)
                round_trunc = max(round_trunc, tx, ty)
                if d > 0:
                    nxlo, nxhi = p.xlo - ys_hi, p.xhi - ys_lo
                    nylo, nyhi = p.ylo + xs_lo, p.yhi + xs_hi
                    nzlo, nzhi = p.zlo - atan_hi, p.zhi - atan_lo
                else:
                    nxlo, nxhi = p.xlo + ys_lo, p.xhi + ys_hi
                    nylo, nyhi = p.ylo - xs_hi, p.yhi - xs_lo
                    nzlo, nzhi = p.zlo + atan_lo, p.zhi + atan_hi
                next_paths.append(_Path(p.sign, p.steps, nxlo, nxhi, nylo, nyhi, nzlo, nzhi))
        # Paths propagate independently: mid-iteration box-merging would
        # destroy the angle correlation between the x/y and z coordinates
        # (two trajectories that diverge transiently after an early split
        # re-converge by the final round).  The union is taken only at the
        # very end.  If the defensive path budget is ever exceeded, same-sign
        # paths are merged into their box union — conservative, never wrong.
        if len(next_paths) > _PATH_BUDGET:
            merged: dict[int, _Path] = {}
            for p in next_paths:
                m = merged.get(p.sign)
                if m is None:
                    merged[p.sign] = p
                else:
                    m.xlo, m.xhi = min(m.xlo, p.xlo), max(m.xhi, p.xhi)
                    m.ylo, m.yhi = min(m.ylo, p.ylo), max(m.yhi, p.yhi)
                    m.zlo, m.zhi = min(m.zlo, p.zlo), max(m.zhi, p.zhi)
                    m.steps = max(m.steps, p.steps)
            next_paths = list(merged.values())
        paths = next_paths
        for p in paths:
            _check_range(p, f"envelope.round[{i}]")
        if record:
            # Widths of the union over all live paths (single path in practice).
            xlo = min(p.xlo for p in paths)
            xhi = max(p.xhi for p in paths)
            ylo = min(p.ylo for p in paths)
            yhi = max(p.yhi for p in paths)
            zlo = min(p.zlo for p in paths)
            zhi = max(p.zhi for p in paths)
            rounds.append(RoundEnvelope(
                index=i,
                directions=tuple(sorted(round_dirs, reverse=True)),
                x_width=xhi - xlo,
                y_width=yhi - ylo,
                z_width=zhi - zlo,
                shift_trunc=round_trunc,
            ))
    return paths, tuple(rounds)


def _final_intervals(paths: list[_Path]) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    """Fold the residual z back into x/y and apply each path's quadrant sign.

    The exact recurrence ends at angle ``z0 - z32``; the target is ``z0``.
    sin/cos are 1-Lipschitz, so widening by ``max(|z_lo|, |z_hi|)`` covers the
    residual-angle term.
    """
    sin_lo = sin_hi = cos_lo = cos_hi = None
    for p in paths:
        w = max(-p.zlo, p.zhi)
        xlo, xhi = p.xlo - w, p.xhi + w
        ylo, yhi = p.ylo - w, p.yhi + w
        if p.sign > 0:
            slo, shi, clo, chi = ylo, yhi, xlo, xhi
        else:
            slo, shi, clo, chi = -yhi, -ylo, -xhi, -xlo
        if sin_lo is None:
            sin_lo, sin_hi, cos_lo, cos_hi = slo, shi, clo, chi
        else:
            sin_lo = min(sin_lo, slo)
            sin_hi = max(sin_hi, shi)
            cos_lo = min(cos_lo, clo)
            cos_hi = max(cos_hi, chi)
    assert sin_lo is not None and sin_hi is not None and cos_lo is not None and cos_hi is not None
    return sin_lo, sin_hi, cos_lo, cos_hi


def _halfwidth(lo: Fraction, hi: Fraction) -> Fraction:
    return (hi - lo) / 2


def compute_envelope(angle_q: int) -> EnvelopeResult:
    """Compute the conservative sin/cos error envelope for a Q4.28 angle.

    ``angle_q`` obeys exactly the same contract as :func:`sin_cos`: a genuine
    ``int`` (``bool`` rejected) in signed 32-bit range, else
    :class:`InvalidArgumentTypeError` / :class:`FixedPointOverflowError`.
    Overflow during propagation raises :class:`FixedPointOverflowError` rather
    than saturating.
    """
    angle_q = require_int(angle_q, "angle_q")
    if not (INT32_MIN <= angle_q <= INT32_MAX):
        raise FixedPointOverflowError(
            f"angle_q: {angle_q} outside signed 32-bit Q4.28 range "
            f"[{INT32_MIN}, {INT32_MAX}] (radians must lie in [-8.0, 8.0))",
            value=angle_q,
            where="angle_q",
        )

    total_paths, rounds = _propagate(angle_q, gain_err=True, table_err=True,
                                     shift_err=True, record=True)
    sin_lo, sin_hi, cos_lo, cos_hi = _final_intervals(total_paths)

    # The input is one explicit quantized angle, so reduction is
    # deterministic: every live path must carry the same quadrant sign,
    # matching the core's reduction (late-round direction splits only
    # multiply paths within that one sign).
    if len({p.sign for p in total_paths}) != 1:
        raise RangeReductionError(
            "envelope reduction of a point angle produced mixed quadrant signs"
        )
    reduced = angle_q
    while reduced > HALF_PI_Q or reduced < -HALF_PI_Q:
        if reduced > HALF_PI_Q:
            reduced -= PI_Q
        else:
            reduced += PI_Q
    reduced_q = reduced
    sign_flip = total_paths[0].sign

    def isolated(**kw: bool) -> tuple[Fraction, Fraction, Fraction, Fraction]:
        paths, _ = _propagate(angle_q, record=False, **kw)
        return _final_intervals(paths)

    g = isolated(gain_err=True, table_err=False, shift_err=False)
    t = isolated(gain_err=False, table_err=True, shift_err=False)
    s = isolated(gain_err=False, table_err=False, shift_err=True)
    sources = ErrorSources(
        gain_quantization=SourceWidth(sin=_halfwidth(g[0], g[1]), cos=_halfwidth(g[2], g[3])),
        table_quantization=SourceWidth(sin=_halfwidth(t[0], t[1]), cos=_halfwidth(t[2], t[3])),
        shift_truncation=SourceWidth(sin=_halfwidth(s[0], s[1]), cos=_halfwidth(s[2], s[3])),
    )

    floor = lambda v: v.numerator // v.denominator  # noqa: E731 - exact floor
    ceil = lambda v: -((-v.numerator) // v.denominator)  # noqa: E731
    return EnvelopeResult(
        angle_q=angle_q,
        reduced_q=reduced_q,
        sign_flip=sign_flip,
        sin_lo=floor(sin_lo),
        sin_hi=ceil(sin_hi),
        cos_lo=floor(cos_lo),
        cos_hi=ceil(cos_hi),
        sin_lo_exact=sin_lo,
        sin_hi_exact=sin_hi,
        cos_lo_exact=cos_lo,
        cos_hi_exact=cos_hi,
        sources=sources,
        rounds=rounds,
    )

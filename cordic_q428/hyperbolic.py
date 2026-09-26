"""Hyperbolic rotation-mode CORDIC in signed Q4.28 fixed point.

Computes cosh(z) and sinh(z) for a Q4.28 angle z with |z| <= 1, using
hyperbolic micro-rotations starting at i = 1:

    x' = x + d * (y >> i)
    y' = y + d * (x >> i)
    z' = z - d * atanh(2**-i),   d = +1 if z >= 0 else -1

Convergence requires the rounds at i = 4, 13, 40, ... (i_next = 3*i + 1) to
be executed twice; with 32 iterations the repeated indices are (4, 13), so
the schedule has 34 micro-rotations.  The inverse-gain constant
(:data:`cordic_q428.tables.HYP_GAIN_INV_Q`) is the product of
1/sqrt(1 - 2**-2i) over every executed micro-rotation, repeats included, so
the duplicated rounds are compensated exactly.

Contract highlights (mirrors the circular core):

* Input domain is |z| <= 1 exactly (|z_q| <= SCALE); anything outside raises
  :class:`InputRangeError` -- rejected, never clamped or saturated.
* ``bool`` is rejected as an integer argument.
* Every right shift is arithmetic, rounding toward negative infinity
  (:func:`cordic_q428.fixedpoint.sar`); each round records the actual shift
  amount and the shifted operands that were used, so the rounding applied is
  fully traceable.
* Every register update is range-checked against signed 32-bit; any overflow
  raises :class:`FixedPointOverflowError`.  Nothing wraps, nothing saturates.
* Errors are reported against the independent Decimal exponential reference
  :func:`cordic_q428.reference.sinh_cosh_ref`.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext

from .errors import InputRangeError
from .fixedpoint import INT32_MAX, INT32_MIN, SCALE, check_int32, require_int, sar
from .reference import sinh_cosh_ref
from .tables import ATANH_TABLE_Q, HYP_GAIN_INV_Q, HYP_SCHEDULE

# Accepted input domain: |z| <= 1 exactly, i.e. |z_q| <= SCALE in Q4.28.
Z_MAX_Q: int = SCALE


@dataclass(frozen=True)
class HyperbolicRoundState:
    """State of the hyperbolic CORDIC datapath after one micro-rotation.

    ``index`` is the CORDIC shift index i (1-based, as hyperbolic rotation
    starts at i = 1).  ``repeated`` is True for the duplicated micro-rotation
    at the convergence-required repeat indices (4, 13).  ``shift`` is the
    actual right-shift amount applied (always equal to ``index`` here, kept
    explicit so the trace records the real datapath operation), and
    ``x_shift``/``y_shift`` are the actual shifted operands used, i.e.
    ``sar(x, shift)``/``sar(y, shift)`` of the *old* registers -- the rounding
    introduced by the shift is exactly the difference from the unshifted
    value.  ``direction`` is +1 or -1 (sign of z before the update).
    """

    round_no: int
    index: int
    repeated: bool
    shift: int
    direction: int
    x_shift: int
    y_shift: int
    x: int
    y: int
    z: int


@dataclass(frozen=True)
class HyperbolicResult:
    """Result of :func:`sinh_cosh`.

    ``cosh_q``/``sinh_q`` are the Q4.28 integer outputs.  ``rounds`` holds the
    per-micro-rotation states (34 entries for 32 iterations with the repeats
    at i = 4 and i = 13).  ``cosh_error``/``sinh_error`` are Decimals:
    ``output/2**28 - reference`` in real units, against the independent
    Decimal exponential oracle.
    """

    cosh_q: int
    sinh_q: int
    z_q: int
    rounds: tuple[HyperbolicRoundState, ...]
    cosh_error: Decimal
    sinh_error: Decimal

    @property
    def cosh(self) -> float:
        return self.cosh_q / SCALE

    @property
    def sinh(self) -> float:
        return self.sinh_q / SCALE


def _validate_z_q(z_q: object) -> int:
    z_q = require_int(z_q, "z_q")
    if not (INT32_MIN <= z_q <= INT32_MAX):
        raise InputRangeError(
            f"z_q: {z_q} outside signed 32-bit Q4.28 range "
            f"[{INT32_MIN}, {INT32_MAX}]",
            value=z_q,
            where="z_q",
        )
    if abs(z_q) > Z_MAX_Q:
        raise InputRangeError(
            f"z_q: |z| must be <= 1 for the hyperbolic rotation entry "
            f"(|z_q| <= {Z_MAX_Q}), got {z_q}",
            value=z_q,
            where="z_q",
        )
    return z_q


def sinh_cosh(z_q: int) -> HyperbolicResult:
    """Compute cosh(z) and sinh(z) of a Q4.28 angle z, |z| <= 1.

    Runs the hyperbolic rotation-mode CORDIC over the frozen schedule
    (i = 1..32 with the rounds at i = 4 and i = 13 repeated).  Pure function:
    a rejected input leaves no partial state.
    """
    z_q = _validate_z_q(z_q)

    x = HYP_GAIN_INV_Q
    y = 0
    z = z_q
    rounds: list[HyperbolicRoundState] = []
    for round_no, (i, repeated) in enumerate(HYP_SCHEDULE):
        d = 1 if z >= 0 else -1
        # Arithmetic shift right, rounding toward -inf, on the *old* values.
        x_shift = sar(x, i)
        y_shift = sar(y, i)
        if d > 0:
            x_new = check_int32(x + y_shift, f"round[{round_no}].x")
            y_new = check_int32(y + x_shift, f"round[{round_no}].y")
            z_new = check_int32(z - ATANH_TABLE_Q[i], f"round[{round_no}].z")
        else:
            x_new = check_int32(x - y_shift, f"round[{round_no}].x")
            y_new = check_int32(y - x_shift, f"round[{round_no}].y")
            z_new = check_int32(z + ATANH_TABLE_Q[i], f"round[{round_no}].z")
        x, y, z = x_new, y_new, z_new
        rounds.append(
            HyperbolicRoundState(
                round_no=round_no,
                index=i,
                repeated=repeated,
                shift=i,
                direction=d,
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

    return HyperbolicResult(
        cosh_q=cosh_q,
        sinh_q=sinh_q,
        z_q=z_q,
        rounds=tuple(rounds),
        cosh_error=+cosh_err,
        sinh_error=+sinh_err,
    )

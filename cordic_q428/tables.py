"""High-precision constant tables for the CORDIC core, generated with Decimal.

Everything in this module is derived at import time from first principles
using only the standard-library ``decimal`` module:

* ``PI_DEC`` — pi from the Machin formula ``16*atan(1/5) - 4*atan(1/239)``,
  where each ``atan`` is its Taylor series evaluated at 80-digit precision.
* ``ATAN_TABLE_Q`` — ``atan(2**-i)`` for ``i = 0..31`` (``atan(1)`` is taken
  as ``pi/4`` because the raw Taylor series at ``x = 1`` converges too
  slowly), each quantized to Q4.28 with round-half-to-even.
* ``GAIN_INV_Q`` — the inverse CORDIC gain ``prod(1/sqrt(1 + 2**-2i))`` over
  the 32 iteration indices, quantized to Q4.28 the same way.

Quantization uses ``ROUND_HALF_EVEN`` (ties-to-even) exactly as specified.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal, localcontext

from .fixedpoint import INT32_MIN, INT32_MAX, SCALE, check_int32

ITERATIONS: int = 32
_TABLE_PREC: int = 80


def _atan_series(x: Decimal, prec: int) -> Decimal:
    """Taylor series atan(x) = sum (-1)^k x^(2k+1)/(2k+1), for |x| <= 0.5."""
    with localcontext() as ctx:
        ctx.prec = prec
        x2 = x * x
        term = x
        total = x
        eps = Decimal(10) ** (-(prec - 10))
        k = 1
        while True:
            term *= x2
            add = term / (2 * k + 1)
            if k % 2:
                total -= add
            else:
                total += add
            if abs(add) < eps:
                return +total
            k += 1


def _compute_pi(prec: int) -> Decimal:
    """Machin formula: pi = 16*atan(1/5) - 4*atan(1/239)."""
    with localcontext() as ctx:
        ctx.prec = prec
        pi = 16 * _atan_series(Decimal(1) / 5, prec) - 4 * _atan_series(Decimal(1) / 239, prec)
        return +pi


def _to_q(value: Decimal, where: str) -> int:
    """Quantize a Decimal to Q4.28 with round-half-to-even (ties-to-even)."""
    with localcontext() as ctx:
        ctx.prec = _TABLE_PREC
        scaled = value * SCALE
        q = int(scaled.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))
    return check_int32(q, where)


def _build_tables() -> tuple[Decimal, tuple[int, ...], int]:
    with localcontext() as ctx:
        ctx.prec = _TABLE_PREC
        pi = _compute_pi(_TABLE_PREC)
        atan_entries: list[int] = []
        for i in range(ITERATIONS):
            if i == 0:
                angle = pi / 4  # atan(1); the direct series at x=1 converges too slowly
            else:
                angle = _atan_series(Decimal(1) / (1 << i), _TABLE_PREC)
            atan_entries.append(_to_q(angle, f"atan_table[{i}]"))
        gain_inv = Decimal(1)
        for i in range(ITERATIONS):
            factor = (Decimal(1) + Decimal(2) ** (-2 * i)).sqrt()
            gain_inv /= factor
        gain_inv_q = _to_q(gain_inv, "gain_inverse")
        return +pi, tuple(atan_entries), gain_inv_q


PI_DEC, ATAN_TABLE_Q, GAIN_INV_Q = _build_tables()

# Quantized circle constants used by the quadrant reduction, all derived from
# the same high-precision pi and quantized ties-to-even.
PI_Q: int = _to_q(PI_DEC, "pi")
HALF_PI_Q: int = _to_q(PI_DEC / 2, "pi/2")
TWO_PI_Q: int = _to_q(PI_DEC * 2, "2*pi")

# Sanity invariants checked at import time: a broken table must fail loudly.
assert len(ATAN_TABLE_Q) == ITERATIONS
assert all(INT32_MIN <= v <= INT32_MAX for v in ATAN_TABLE_Q)
# Non-increasing: for i >= 29, atan(2**-i) quantizes to 0 in Q4.28.
assert all(ATAN_TABLE_Q[i] >= ATAN_TABLE_Q[i + 1] for i in range(ITERATIONS - 1))


# ---------------------------------------------------------------------------
# Hyperbolic mode (rotation): cosh/sinh via atanh(2**-i) micro-rotations.
#
# Hyperbolic CORDIC starts at i = 1 (atanh(1) is infinite) and must *repeat*
# the iterations at i = 4, 13, 40, ... (i_next = 3*i + 1) to converge; the
# gain product below therefore includes those repeated rounds exactly once
# extra each.  HYPER_ITERATIONS = 40 so the i = 40 repeat is included.
# ---------------------------------------------------------------------------

HYPER_ITERATIONS: int = 40


def _hyper_repeat_indices(n: int) -> tuple[int, ...]:
    """Indices i in [1, n] that must be executed twice: 4, 13, 40, ..."""
    reps: list[int] = []
    i = 4
    while i <= n:
        reps.append(i)
        i = 3 * i + 1
    return tuple(reps)


HYPER_REPEAT_INDICES: tuple[int, ...] = _hyper_repeat_indices(HYPER_ITERATIONS)


def _atanh_series(x: Decimal, prec: int) -> Decimal:
    """Taylor series atanh(x) = sum x**(2k+1)/(2k+1), converges for |x| < 1."""
    with localcontext() as ctx:
        ctx.prec = prec
        x2 = x * x
        term = x
        total = x
        eps = Decimal(10) ** (-(prec - 10))
        k = 1
        while True:
            term *= x2
            add = term / (2 * k + 1)
            total += add
            if abs(add) < eps:
                return +total
            k += 1


def _build_hyper_tables() -> tuple[tuple[int, ...], int]:
    with localcontext() as ctx:
        ctx.prec = _TABLE_PREC
        entries: list[int] = []
        for i in range(1, HYPER_ITERATIONS + 1):
            angle = _atanh_series(Decimal(1) / (1 << i), _TABLE_PREC)
            entries.append(_to_q(angle, f"atanh_table[{i}]"))
        gain_inv = Decimal(1)
        executed: list[int] = list(range(1, HYPER_ITERATIONS + 1))
        executed += [i for i in HYPER_REPEAT_INDICES]
        for i in executed:
            factor = (Decimal(1) - Decimal(2) ** (-2 * i)).sqrt()
            gain_inv /= factor
        gain_inv_q = _to_q(gain_inv, "hyper_gain_inverse")
        return tuple(entries), gain_inv_q


# ATANH_TABLE_Q[k] is atanh(2**-(k+1)) quantized to Q4.28 ties-to-even, so
# iteration index i (1-based) uses ATANH_TABLE_Q[i - 1].
ATANH_TABLE_Q, GAIN_HYP_INV_Q = _build_hyper_tables()

# Import-time invariants: a broken hyperbolic table must fail loudly.
assert len(ATANH_TABLE_Q) == HYPER_ITERATIONS
assert all(INT32_MIN <= v <= INT32_MAX for v in ATANH_TABLE_Q)
assert all(ATANH_TABLE_Q[i] >= ATANH_TABLE_Q[i + 1] for i in range(HYPER_ITERATIONS - 1))
assert HYPER_REPEAT_INDICES == (4, 13, 40)

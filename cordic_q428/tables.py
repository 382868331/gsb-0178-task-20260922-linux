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
* ``GAIN_PREFIX_INV_Q`` — ``GAIN_PREFIX_INV_Q[m]`` is the inverse gain of the
  first ``m`` rounds only, each prefix product quantized once.  Vectoring
  mode uses it when an exactly axis-aligned ``y`` ends the micro-rotations
  before round 32; ``GAIN_PREFIX_INV_Q[32] == GAIN_INV_Q``.

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


def _build_tables() -> tuple[Decimal, tuple[int, ...], int, tuple[int, ...]]:
    with localcontext() as ctx:
        ctx.prec = _TABLE_PREC
        pi = _compute_pi(_TABLE_PREC)
        atan_entries: list[int] = []
        factors: list[Decimal] = []
        for i in range(ITERATIONS):
            if i == 0:
                angle = pi / 4  # atan(1); the direct series at x=1 converges too slowly
            else:
                angle = _atan_series(Decimal(1) / (1 << i), _TABLE_PREC)
            atan_entries.append(_to_q(angle, f"atan_table[{i}]"))
            factors.append((Decimal(1) + Decimal(2) ** (-2 * i)).sqrt())
        # Prefix products of the inverse gain: prefix[k] = prod over rounds
        # 0..k-1 of 1/sqrt(1 + 4**-i), each quantized once.  Vectoring mode
        # compensates with prefix[rotated_rounds] because an axis-aligned
        # vector stops performing micro-rotations.  prefix[ITERATIONS] is the
        # ordinary full-gain inverse used by rotation mode.
        prefix_q: list[int] = []
        prod = Decimal(1)
        for k in range(ITERATIONS + 1):
            prefix_q.append(_to_q(1 / prod if k else Decimal(1), f"gain_prefix[{k}]"))
            if k < ITERATIONS:
                prod *= factors[k]
        gain_inv_q = prefix_q[ITERATIONS]
        return +pi, tuple(atan_entries), gain_inv_q, tuple(prefix_q)


PI_DEC, ATAN_TABLE_Q, GAIN_INV_Q, GAIN_PREFIX_INV_Q = _build_tables()

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
# Prefix gain table: 33 entries, starting at 1, strictly decreasing, ending
# at the full inverse gain.
assert len(GAIN_PREFIX_INV_Q) == ITERATIONS + 1
assert GAIN_PREFIX_INV_Q[0] == SCALE
assert GAIN_PREFIX_INV_Q[ITERATIONS] == GAIN_INV_Q
# Non-increasing: the last few per-round differences fall below 1 ulp.
assert all(GAIN_PREFIX_INV_Q[m] >= GAIN_PREFIX_INV_Q[m + 1] for m in range(ITERATIONS))

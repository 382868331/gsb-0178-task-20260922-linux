"""A-priori error bounds for early termination of the fixed-point CORDIC.

The bound in this module is a *proven* worst-case bound on the absolute
error of each output coordinate (sin and cos of the *quantized Q4.28 input
angle*) when the rotation-mode CORDIC is stopped after ``n`` of the 32
rounds, ``1 <= n <= ITERATIONS``.  It never compares two successive
iterations: a small step-to-step change says nothing about the absolute
error because the residual angle and the fixed-point truncation are not
visible from the step delta.

Bound components, all measured in Q4.28 ulp of the x/y coordinates:

1. **Residual angle.** After ``n`` greedy micro-rotations the un-rotated
   angle of the exact-table trajectory obeys the standard CORDIC residual
   theorem ``|theta_n| <= atan(2**-(n-1))``.  Each table entry is quantized
   ties-to-even with ``|epsilon_i| <= 0.5`` ulp, and the accumulated
   ``n/2`` ulp slack also covers trajectories whose decisions differ from
   the exact-trajectory decisions near a sign flip.  Rotating a vector of
   norm ``<= 1`` by ``|theta_n|`` perturbs either coordinate by at most
   ``|theta_n|`` real.  The datapath is initialized with the *32-step*
   inverse gain, so after ``n`` rounds the vector norm is at most
   ``K_n/K_32 <= 1``.
2. **Gain mismatch.** Stopping early leaves the tail gain
   ``G_n = prod_{j=n}^{31} sqrt(1 + 2**(-2j))`` uncancelled, a relative
   magnitude error of at most ``G_n - 1`` and hence an absolute error of at
   most ``G_n - 1`` (norm ``<= 1``).
3. **Shift truncation.** Each round ``x >> i``/``y >> i`` injects an error
   vector of Euclidean norm ``<= sqrt(2)`` ulp.  A later micro-rotation
   multiplies vector norms by exactly ``f_j = sqrt(1 + 2**(-2j))``.  The
   summed propagation factor obeys ``A_0 = 0``,
   ``A_n = 1 + f_{n-1} * A_{n-1}``.
4. **Initial gain quantization.** ``|x0 - 1/K_32| <= 0.5`` ulp in the
   register; the micro-rotations multiply its norm by ``K_n`` after ``n``
   rounds.

Total bound in ulp::

    B_n = (K_n/K_32) * (atan(2**-(n-1)) * SCALE + n/2)
          + (G_n - 1) * SCALE + sqrt(2) * A_n + 0.5 * K_n

The table is precomputed at 80-digit Decimal precision, so the bound is a
guarantee, not an estimate.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

from .fixedpoint import SCALE
from .tables import ATAN_TABLE_DEC, ITERATIONS

_BOUND_PREC: int = 80


def _build_bounds() -> tuple[tuple[Decimal, ...], Decimal]:
    """Precompute ``(B_1, ..., B_32)`` in real units and their minimum."""
    with localcontext() as ctx:
        ctx.prec = _BOUND_PREC
        scale = Decimal(SCALE)
        factors = [
            (Decimal(1) + Decimal(2) ** (-2 * j)).sqrt()
            for j in range(ITERATIONS)
        ]

        # Cumulative gains: K_n = prod_{j=0}^{n-1} f_j, K_0 = 1.
        k_gain = [Decimal(1)] * (ITERATIONS + 1)
        for j in range(ITERATIONS):
            k_gain[j + 1] = k_gain[j] * factors[j]

        # Uncancelled tail gains: G_n = prod_{j=n}^{31} f_j, G_32 = 1.
        g_tail = [Decimal(1)] * (ITERATIONS + 1)
        for j in range(ITERATIONS - 1, -1, -1):
            g_tail[j] = g_tail[j + 1] * factors[j]

        # Summed truncation propagation: A_0 = 0, A_n = 1 + f_{n-1} A_{n-1}.
        trunc = [Decimal(0)] * (ITERATIONS + 1)
        for n in range(1, ITERATIONS + 1):
            trunc[n] = Decimal(1) + factors[n - 1] * trunc[n - 1]

        sqrt2 = Decimal(2).sqrt()
        k_last = k_gain[ITERATIONS]
        bounds: list[Decimal] = []
        for n in range(1, ITERATIONS + 1):
            norm_scale = k_gain[n] / k_last  # K_n/K_32 <= 1
            residual_angle_ulp = ATAN_TABLE_DEC[n - 1] * scale + Decimal(n) / 2
            bound_ulp = (
                norm_scale * residual_angle_ulp
                + (g_tail[n] - Decimal(1)) * scale
                + sqrt2 * trunc[n]
                + Decimal("0.5") * k_gain[n]
            )
            bounds.append(+bound_ulp / scale)

        best = min(bounds)
        return tuple(bounds), +best


def _build_min() -> tuple[Decimal, int]:
    best_index = min(range(ITERATIONS), key=lambda i: ERROR_BOUNDS_REAL[i])
    with localcontext() as ctx:
        ctx.prec = _BOUND_PREC
        return +ERROR_BOUNDS_REAL[best_index], best_index + 1


ERROR_BOUNDS_REAL: tuple[Decimal, ...] = _build_bounds()[0]
MIN_GUARANTEED_ERROR_REAL: Decimal
MIN_GUARANTEED_ERROR_STEPS: int
MIN_GUARANTEED_ERROR_REAL, MIN_GUARANTEED_ERROR_STEPS = _build_min()


def error_bound_real(steps: int) -> Decimal:
    """Return the guaranteed absolute error bound (real units) after ``steps``.

    ``steps`` must satisfy ``1 <= steps <= ITERATIONS``.
    """
    if not isinstance(steps, int) or isinstance(steps, bool):
        raise TypeError(f"steps: expected int, got {type(steps).__name__}")
    if not (1 <= steps <= ITERATIONS):
        raise ValueError(f"steps: {steps} outside [1, {ITERATIONS}]")
    with localcontext() as ctx:
        ctx.prec = _BOUND_PREC
        return +ERROR_BOUNDS_REAL[steps - 1]


def error_bound_ulp(steps: int) -> Decimal:
    """Return the guaranteed absolute error bound in Q4.28 ulp after ``steps``."""
    with localcontext() as ctx:
        ctx.prec = _BOUND_PREC
        return +(error_bound_real(steps) * SCALE)


__all__ = [
    "ERROR_BOUNDS_REAL",
    "MIN_GUARANTEED_ERROR_REAL",
    "MIN_GUARANTEED_ERROR_STEPS",
    "error_bound_real",
    "error_bound_ulp",
]

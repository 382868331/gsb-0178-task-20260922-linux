"""Deterministic demo for the Q4.28 fixed-point CORDIC library.

Runs entirely offline with fixed inputs (no randomness, no network):

1. One normal sin/cos computation, printed with its per-round trace summary
   and its error against the independent Decimal reference.
2. A deterministic sweep over the full representable angle range reporting
   the maximum observed error (with an explicit non-exhaustive note).
3. One genuinely triggered rejection (out-of-range angle) and one triggered
   type rejection (bool), printed as caught exceptions.
"""

import time

from cordic_q428 import (
    INT32_MAX,
    INT32_MIN,
    MIN_GUARANTEED_ERROR_REAL,
    SCALE,
    CordicError,
    float_to_q,
    sin_cos,
    sin_cos_radians,
    ToleranceNotGuaranteedError,
)


def main() -> None:
    t0 = time.perf_counter()

    print("=== 1. Normal result: sin/cos(1.0 rad) in signed Q4.28 ===")
    r = sin_cos_radians(1.0)
    print(f"  angle_q   = {float_to_q(1.0)}  (1.0 rad as Q4.28)")
    print(f"  sin_q     = {r.sin_q}  -> {r.sin:.10f}")
    print(f"  cos_q     = {r.cos_q}  -> {r.cos:.10f}")
    print(f"  sin_error = {float(r.sin_error):+.3e}  ({float(r.sin_error) * SCALE:+.3f} ulp)")
    print(f"  cos_error = {float(r.cos_error):+.3e}  ({float(r.cos_error) * SCALE:+.3f} ulp)")
    print(f"  rounds    = {len(r.rounds)} iterations, "
          f"final z residual = {r.rounds[-1].z} ulp "
          f"({r.rounds[-1].z / SCALE:+.3e} rad)")

    print()
    print("=== 2. Deterministic full-range sweep (fixed grid, 2001 points) ===")
    n = 2001
    worst_ulp = 0.0
    worst_q = 0
    for k in range(n):
        q = INT32_MIN + k * (2**32 - 1) // (n - 1)
        res = sin_cos(q)
        err = max(abs(float(res.sin_error)), abs(float(res.cos_error))) * SCALE
        if err > worst_ulp:
            worst_ulp, worst_q = err, q
    print(f"  max observed |error| = {worst_ulp:.3f} ulp "
          f"({worst_ulp / SCALE:.3e} abs) at angle_q = {worst_q}")
    print(f"  NOTE: the sweep samples {n} of 2**32 = 4294967296 representable")
    print("  angles; it is NOT exhaustive. The bound above is observed, not proven.")

    print()
    print("=== 3. Early termination with a certified error tolerance ===")
    for tol in (1e-4, 1e-5, 1e-6):
        r = sin_cos_radians(1.0, max_error=tol)
        actual = max(abs(float(r.sin_error)), abs(float(r.cos_error))) * SCALE
        print(f"  max_error={tol:.0e}: executed {r.steps_executed:2d}/32 rounds, "
              f"guarantee_met={r.tolerance_met}, "
              f"certified bound={float(r.error_bound) * SCALE:.2f} ulp, "
              f"actual error={actual:.2f} ulp")
    too_tight = 1e-9
    print(f"  max_error={too_tight:.0e} is below the Q4.28 resolution:")
    try:
        sin_cos_radians(1.0, max_error=too_tight)
    except ToleranceNotGuaranteedError as exc:
        print(f"    -> {type(exc).__name__}: smallest provable bound is "
              f"{float(exc.best_bound):.3e} ({float(exc.best_bound) * SCALE:.1f} ulp) "
              f"at {exc.steps} rounds; the tolerance is NOT relaxed")
    print(f"  (smallest attainable certified bound in this format: "
          f"{float(MIN_GUARANTEED_ERROR_REAL):.3e})")

    print()
    print("=== 4. Triggered rejections (invalid input is refused, not corrected) ===")
    try:
        sin_cos_radians(8.0)  # Q4.28 upper bound is 8.0 - 1 ulp
    except CordicError as exc:
        print(f"  angle 8.0 rad -> rejected: {type(exc).__name__}: {exc}")
    try:
        sin_cos(True)  # bool is not a valid fixed-point integer
    except CordicError as exc:
        print(f"  angle True    -> rejected: {type(exc).__name__}: {exc}")
    try:
        sin_cos(INT32_MAX + 1)  # outside signed 32-bit
    except CordicError as exc:
        print(f"  angle 2**31   -> rejected: {type(exc).__name__}: {exc}")

    print()
    print(f"demo completed in {time.perf_counter() - t0:.2f}s (real computation, no sleep)")


if __name__ == "__main__":
    main()

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
    SCALE,
    CordicError,
    ToleranceNotGuaranteedError,
    float_to_q,
    sin_cos,
    sin_cos_radians,
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
    print("=== 3. Caller-specified tolerance: early stop vs explicit failure ===")
    relaxed = sin_cos_radians(1.0, tol_ulp=300)
    print(f"  tol_ulp=300: stopped after {relaxed.steps} of 32 rounds "
          f"(early={relaxed.early_stopped}), tolerance_met={relaxed.tolerance_met}")
    print(f"               proven bound = {float(relaxed.error_bound_ulp):.2f} ulp; "
          f"actual sin/cos error = "
          f"{float(relaxed.sin_error) * SCALE:+.2f}/"
          f"{float(relaxed.cos_error) * SCALE:+.2f} ulp")
    strict = sin_cos_radians(1.0)  # no tolerance: original full 32-step run
    print(f"  default:     {strict.steps} rounds (full computation preserved)")
    try:
        sin_cos_radians(1.0, tol_ulp=1)  # tighter than Q4.28 can guarantee
    except ToleranceNotGuaranteedError as exc:
        print(f"  tol_ulp=1:   refused after {exc.steps} rounds; tightest provable "
              f"bound = {float(exc.bound * SCALE):.2f} ulp (requirement not relaxed)")

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

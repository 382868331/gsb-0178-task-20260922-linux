"""Deterministic offline demo for the hyperbolic CORDIC extension (cosh/sinh).

Runs entirely offline with fixed inputs, well under 90 seconds:

1. One normal cosh/sinh computation at z = 1.0 with per-round trace summary
   (including the repeated rounds at i = 4, 13, 40) and the error against
   the independent Decimal exponential reference.
2. A deterministic sweep over the full supported range |z| <= 1 reporting
   the maximum observed error (with an explicit non-exhaustive note).
3. Genuinely triggered rejections: |z| > 1, bool input, NaN.
"""

import time

from cordic_q428 import (
    HYPER_ITERATIONS,
    HYPER_REPEAT_INDICES,
    SCALE,
    CordicError,
    float_to_q,
    sinh_cosh,
    sinh_cosh_radians,
)


def main() -> None:
    t0 = time.perf_counter()

    print("=== 1. Normal result: cosh/sinh(1.0) in signed Q4.28 ===")
    r = sinh_cosh(SCALE)
    print(f"  z_q        = {SCALE}  (1.0 as Q4.28)")
    print(f"  sinh_q     = {r.sinh_q}  -> {r.sinh:.10f}")
    print(f"  cosh_q     = {r.cosh_q}  -> {r.cosh:.10f}")
    print(f"  sinh_error = {float(r.sinh_error):+.3e}  ({float(r.sinh_error) * SCALE:+.3f} ulp)")
    print(f"  cosh_error = {float(r.cosh_error):+.3e}  ({float(r.cosh_error) * SCALE:+.3f} ulp)")
    print(f"  rounds     = {len(r.rounds)} micro-rotations "
          f"(i = 1..{HYPER_ITERATIONS}, repeats at i = {HYPER_REPEAT_INDICES})")
    for st in r.rounds:
        if st.repeated or st.index <= 2:
            print(f"    i={st.index:2d}{' (repeat)' if st.repeated else '          '}"
                  f" d={st.direction:+d} shift={st.shift:2d}"
                  f" x>>i={st.x_shift:12d} y>>i={st.y_shift:12d}"
                  f" x={st.x:12d} y={st.y:12d} z={st.z:12d}")

    print()
    print("=== 2. Deterministic sweep over |z| <= 1 (fixed grid, 2001 points) ===")
    n = 2001
    worst_ulp = 0.0
    worst_q = 0
    for k in range(n):
        q = -SCALE + k * (2 * SCALE) // (n - 1)
        res = sinh_cosh(q)
        err = max(abs(float(res.sinh_error)), abs(float(res.cosh_error))) * SCALE
        if err > worst_ulp:
            worst_ulp, worst_q = err, q
    print(f"  max observed |error| = {worst_ulp:.3f} ulp "
          f"({worst_ulp / SCALE:.3e} abs) at z_q = {worst_q} (z = {worst_q / SCALE:.6f})")
    print(f"  NOTE: the sweep samples {n} of {2 * SCALE + 1} representable inputs")
    print("  in |z| <= 1; it is NOT exhaustive. The bound above is observed, not proven.")

    print()
    print("=== 3. Triggered rejections (invalid input is refused, not saturated) ===")
    try:
        sinh_cosh(SCALE + 1)  # |z| > 1.0 is outside the convergence range
    except CordicError as exc:
        print(f"  z = 1.0 + 1 ulp -> rejected: {type(exc).__name__}: {exc}")
    try:
        sinh_cosh(True)  # bool is not a valid fixed-point integer
    except CordicError as exc:
        print(f"  z = True        -> rejected: {type(exc).__name__}: {exc}")
    try:
        sinh_cosh_radians(float("nan"))
    except CordicError as exc:
        print(f"  z = NaN         -> rejected: {type(exc).__name__}: {exc}")

    print()
    print(f"demo completed in {time.perf_counter() - t0:.2f}s "
          "(real computation, no sleep, no network)")


if __name__ == "__main__":
    main()


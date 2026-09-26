"""Deterministic offline demo for the hyperbolic CORDIC entry point.

Runs entirely offline with fixed inputs (no randomness, no network, no
sleep, no prerecorded results), well under 90 seconds:

1. One normal sinh/cosh computation with its per-round trace, highlighting
   the repeated micro-rotations at i = 4 and i = 13 (i_next = 3*i + 1).
2. Measured maximum error against the independent Decimal exponential
   reference over the acceptance configs: zero, both boundaries, and
   neighborhoods of the repeated rounds, plus a deterministic grid.
3. Genuinely triggered rejections (out-of-domain |z| > 1, bool, and a
   value outside signed 32-bit), printed as caught exceptions.
"""

import time

from cordic_q428 import (
    ATANH_TABLE_Q,
    HYP_REPEAT_INDICES,
    HYP_SCHEDULE,
    INT32_MAX,
    SCALE,
    Z_MAX_Q,
    CordicError,
    sinh_cosh,
)


def _configs():
    zs = {0, Z_MAX_Q, -Z_MAX_Q, 1, -1, Z_MAX_Q - 1, -(Z_MAX_Q - 1)}
    acc = 0
    for i, repeated in HYP_SCHEDULE:
        acc += ATANH_TABLE_Q[i]
        if repeated:
            for delta in range(-4, 5):
                for base in (acc, acc - ATANH_TABLE_Q[i]):
                    z = base + delta
                    if abs(z) <= Z_MAX_Q:
                        zs.add(z)
                        zs.add(-z)
    for k in range(1, 200):
        z = k * Z_MAX_Q // 200
        zs.add(z)
        zs.add(-z)
    return sorted(zs)


def main() -> None:
    t0 = time.perf_counter()

    print("=== 1. Normal result: cosh/sinh(0.5) in signed Q4.28 ===")
    z_q = SCALE // 2
    r = sinh_cosh(z_q)
    print(f"  z_q       = {z_q}  (0.5 as Q4.28)")
    print(f"  cosh_q    = {r.cosh_q}  -> {r.cosh:.10f}")
    print(f"  sinh_q    = {r.sinh_q}  -> {r.sinh:.10f}")
    print(f"  cosh_err  = {float(r.cosh_error):+.3e}  ({float(r.cosh_error) * SCALE:+.3f} ulp)")
    print(f"  sinh_err  = {float(r.sinh_error):+.3e}  ({float(r.sinh_error) * SCALE:+.3f} ulp)")
    print(f"  schedule  = {len(r.rounds)} micro-rotations "
          f"(i = 1..32, repeats at i = {HYP_REPEAT_INDICES})")
    print("  per-round trace (index, repeat flag, shift, direction, z residual):")
    for s in r.rounds:
        mark = "  <== repeated round" if s.repeated else ""
        print(f"    round {s.round_no:2d}: i={s.index:2d} rep={int(s.repeated)} "
              f"shift={s.shift:2d} d={s.direction:+d} z={s.z:12d}{mark}")

    print()
    print("=== 2. Measured max error vs Decimal exp reference (acceptance configs) ===")
    configs = _configs()
    worst_ulp = 0.0
    worst_z = 0
    for z in configs:
        res = sinh_cosh(z)
        err = max(abs(float(res.cosh_error)), abs(float(res.sinh_error))) * SCALE
        if err > worst_ulp:
            worst_ulp, worst_z = err, z
    print(f"  configs: 0, z=+-1 (boundaries), +-4 ulp neighborhoods of the")
    print(f"  repeated rounds i={HYP_REPEAT_INDICES}, plus a 400-point grid "
          f"({len(configs)} points total)")
    print(f"  max observed |error| = {worst_ulp:.3f} ulp "
          f"({worst_ulp / SCALE:.3e} abs) at z_q = {worst_z}")
    print("  NOTE: sampled configs, not an exhaustive 2**28 sweep; the figure")
    print("  above is measured, not a proven bound.")

    print()
    print("=== 3. Triggered rejections (refused, never saturated) ===")
    try:
        sinh_cosh(Z_MAX_Q + 1)  # |z| > 1 is outside the contracted domain
    except CordicError as exc:
        print(f"  z_q = 1+1ulp  -> rejected: {type(exc).__name__}: {exc}")
    try:
        sinh_cosh(True)  # bool is not a valid fixed-point integer
    except CordicError as exc:
        print(f"  z_q = True    -> rejected: {type(exc).__name__}: {exc}")
    try:
        sinh_cosh(INT32_MAX)  # representable in int32 but |z| >> 1
    except CordicError as exc:
        print(f"  z_q = 2**31-1 -> rejected: {type(exc).__name__}: {exc}")

    print()
    print(f"demo completed in {time.perf_counter() - t0:.2f}s "
          "(real computation, no sleep, no network)")


if __name__ == "__main__":
    main()

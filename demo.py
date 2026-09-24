"""Deterministic demo for the Q4.28 fixed-point CORDIC library.

Runs entirely offline with fixed inputs (no randomness, no network).  It
exercises both CORDIC modes and prints genuinely computed values, not
pre-recorded answers:

1. Rotation mode: one normal sin/cos result with its error vs the Decimal
   reference, plus a deterministic full-range sweep reporting the maximum
   observed error (with an explicit non-exhaustive note).
2. Vectoring mode: atan2/magnitude for a hand-computed 3-4-5 triple and the
   four quadrants, followed by a deterministic polar-ring sweep reporting
   the worst observed angle/magnitude error.
3. Edge cases: the zero vector, INT32_MIN coordinates, and magnitude
   saturation.
4. Genuinely triggered rejections (out-of-range, bool, NaN), printed as
   caught exceptions.
"""

import math
import time

from cordic_q428 import (
    INT32_MAX,
    INT32_MIN,
    PI_Q,
    SCALE,
    CordicError,
    atan2_mag,
    atan2_mag_radians,
    float_to_q,
    sin_cos,
    sin_cos_radians,
)


def main() -> None:
    t0 = time.perf_counter()

    print("=== 1. Rotation mode: sin/cos(1.0 rad) in signed Q4.28 ===")
    r = sin_cos_radians(1.0)
    print(f"  angle_q   = {float_to_q(1.0)}  (1.0 rad as Q4.28)")
    print(f"  sin_q     = {r.sin_q}  -> {r.sin:.10f}")
    print(f"  cos_q     = {r.cos_q}  -> {r.cos:.10f}")
    print(f"  sin_error = {float(r.sin_error):+.3e}  ({float(r.sin_error) * SCALE:+.3f} ulp)")
    print(f"  cos_error = {float(r.cos_error):+.3e}  ({float(r.cos_error) * SCALE:+.3f} ulp)")

    print()
    print("=== 2. Rotation-mode sweep (fixed grid, 2001 angles) ===")
    n = 2001
    worst_ulp = 0.0
    for k in range(n):
        q = INT32_MIN + k * (2**32 - 1) // (n - 1)
        res = sin_cos(q)
        worst_ulp = max(
            worst_ulp,
            abs(float(res.sin_error)) * SCALE,
            abs(float(res.cos_error)) * SCALE,
        )
    print(f"  max observed |error| = {worst_ulp:.3f} ulp ({worst_ulp / SCALE:.3e} abs)")
    print("  NOTE: a fixed grid sample of 2**32 angles, NOT exhaustive;")
    print("  the number is observed, not a proven bound.")

    print()
    print("=== 3. Vectoring mode: atan2/magnitude, hand-computed 3-4-5 ===")
    y3, x4 = 3 * SCALE // 10, 4 * SCALE // 10  # (0.3, 0.4): angle atan(3/4), mag 0.5
    v = atan2_mag(y3, x4)
    print(f"  (y, x) = (0.3, 0.4)")
    print(f"  angle   = {v.angle_q} -> {v.angle:.10f} rad  (atan(3/4) = {math.atan2(0.3, 0.4):.10f})")
    print(f"  magnitude = {v.mag_q} -> {v.mag:.10f}  (expected exactly 0.5)")
    print(f"  errors  = {float(v.angle_error) * SCALE:+.3f} ulp angle, "
          f"{float(v.mag_error) * SCALE:+.3f} ulp magnitude")
    print("  four quadrants at |y|=|x|=1 (diagonals, expected +/- pi/4, +/- 3pi/4):")
    for yy, xx in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        vv = atan2_mag(yy * SCALE, xx * SCALE)
        print(f"    (y={yy:+d}, x={xx:+d}) -> angle {vv.angle:+.9f}, mag {vv.mag:.6f}")

    print()
    print("=== 4. Vectoring-mode sweep (20 polar rings x 401 angles) ===")
    radii = (1, 2, 5, 10, 100, 1000, 10000, 100000,
             SCALE // 4, SCALE // 2, SCALE, 3 * SCALE // 2,
             2 * SCALE, 3 * SCALE, 4 * SCALE, 5 * SCALE, 6 * SCALE,
             7 * SCALE, INT32_MAX - 1, INT32_MAX)
    wa = wm = 0.0
    nsat = 0
    for radius in radii:
        for k in range(401):
            theta = -math.pi + 2 * math.pi * k / 400
            y = round(math.sin(theta) * radius)
            x = round(math.cos(theta) * radius)
            res = atan2_mag(y, x)
            wa = max(wa, abs(float(res.angle_error)) * SCALE)
            if res.saturated:
                nsat += 1
            else:
                wm = max(wm, abs(float(res.mag_error)) * SCALE)
    print(f"  max observed angle error     = {wa:.3f} ulp")
    print(f"  max observed magnitude error = {wm:.3f} ulp (over {8020 - nsat} unsaturated points)")
    print(f"  saturated points = {nsat} (magnitude above the Q4.28 ceiling);")
    print("  NOTE: sampled rings, NOT exhaustive; observed, not proven.")

    print()
    print("=== 5. Edge cases: zero vector, INT32_MIN, saturation ===")
    z = atan2_mag(0, 0)
    print(f"  zero vector (0, 0)      -> angle {z.angle:+.6f}, mag {z.mag:.6f}, "
          f"saturated={z.saturated}, idle rounds={sum(s.direction == 0 for s in z.rounds)}")
    neg = atan2_mag(0, INT32_MIN)
    print(f"  (0, INT32_MIN)          -> angle {neg.angle:+.9f} (convention: +pi), "
          f"mag {neg.mag:.6f}, saturated={neg.saturated}")
    cor = atan2_mag(INT32_MIN, INT32_MIN)
    print(f"  (INT32_MIN, INT32_MIN)  -> angle {cor.angle:+.9f} (-3pi/4), "
          f"mag_q={cor.mag_q} (=INT32_MAX), saturated={cor.saturated}")
    diag = atan2_mag(7 * SCALE, 7 * SCALE)
    print(f"  (7.0, 7.0) mag=sqrt98  -> angle {diag.angle:+.6f} (pi/4), mag saturated "
          f"at {diag.mag_q}, saturated={diag.saturated}")

    print()
    print("=== 6. Triggered rejections (invalid input refused, not corrected) ===")
    try:
        atan2_mag_radians(0.0, 8.0)  # Q4.28 upper bound is 8.0 - 1 ulp
    except CordicError as exc:
        print(f"  x=8.0 rad     -> rejected: {type(exc).__name__}: {exc}")
    try:
        atan2_mag(True, 1)  # bool is not a valid fixed-point integer
    except CordicError as exc:
        print(f"  y=True        -> rejected: {type(exc).__name__}: {exc}")
    try:
        atan2_mag_radians(1.0, math.nan)  # NaN coordinate
    except CordicError as exc:
        print(f"  x=NaN         -> rejected: {type(exc).__name__}: {exc}")
    try:
        atan2_mag(0, INT32_MIN - 1)  # outside signed 32-bit
    except CordicError as exc:
        print(f"  x=INT32_MIN-1 -> rejected: {type(exc).__name__}: {exc}")

    print()
    print(f"demo completed in {time.perf_counter() - t0:.2f}s (real computation, no sleep)")


if __name__ == "__main__":
    main()

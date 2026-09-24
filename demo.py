"""Deterministic demo for the Q4.28 fixed-point CORDIC library.

Runs entirely offline with fixed inputs (no randomness, no network):

Part A — rotation mode (sin/cos), unchanged:
1. One normal sin/cos computation with its error against the Decimal reference.
2. A deterministic full-range sweep reporting the maximum observed error.
3. Genuinely triggered rejections (out-of-range / bool).

Part B — vectoring mode (atan2 + magnitude), new:
4. Hand-computed analytic cases: the four axes, the 45-degree diagonals and a
   3-4-5 triangle.
5. Boundary/defined cases: the zero vector and the most-negative corner
   (saturation), plus the smallest non-zero vectors.
6. A deterministic radius/angle enumeration reporting the worst observed
   error, and genuinely triggered invalid-input errors.
"""

import math
import time

from cordic_q428 import (
    HALF_PI_Q,
    INT32_MAX,
    INT32_MIN,
    PI_Q,
    SCALE,
    CordicError,
    float_to_q,
    polar,
    polar_xy,
    sin_cos,
    sin_cos_radians,
)


def part_a() -> None:
    print("=== A1. Rotation mode: sin/cos(1.0 rad) in signed Q4.28 ===")
    r = sin_cos_radians(1.0)
    print(f"  sin = {r.sin:.10f}  ({float(r.sin_error) * SCALE:+.3f} ulp)")
    print(f"  cos = {r.cos:.10f}  ({float(r.cos_error) * SCALE:+.3f} ulp)")

    print()
    print("=== A2. Full-range sin/cos sweep (fixed grid, 801 points; not exhaustive) ===")
    worst = 0.0
    for k in range(801):
        q = INT32_MIN + k * (2**32 - 1) // 800
        res = sin_cos(q)
        worst = max(
            worst, abs(float(res.sin_error)) * SCALE, abs(float(res.cos_error)) * SCALE
        )
    print(f"  max observed |error| = {worst:.3f} ulp (sampled, not 2**32-exhaustive)")

    print()
    print("=== A3. Rotation-mode rejections ===")
    for label, fn in (
        ("angle 8.0 rad", lambda: sin_cos_radians(8.0)),
        ("angle True", lambda: sin_cos(True)),
        ("angle 2**31", lambda: sin_cos(INT32_MAX + 1)),
    ):
        try:
            fn()
        except CordicError as exc:
            print(f"  {label:14s} -> {type(exc).__name__}")


def part_b() -> None:
    print()
    print("=== B1. Vectoring mode: analytic atan2/magnitude cases ===")
    analytic = (
        ("+x axis (1,0)", (SCALE, 0), 1.0, 0.0),
        ("+y axis (0,1)", (0, SCALE), 1.0, math.pi / 2),
        ("-x axis (-1,0)", (-SCALE, 0), 1.0, math.pi),
        ("-y axis (0,-1)", (0, -SCALE), 1.0, -math.pi / 2),
        ("45 deg (1,1)", (SCALE, SCALE), math.sqrt(2), math.pi / 4),
        ("3-4-5 (3,4)", (3 * SCALE, 4 * SCALE), 5.0, math.atan2(4, 3)),
    )
    for label, (x, y), want_mag, want_ang in analytic:
        res = polar(x, y)
        print(
            f"  {label:18s} mag={res.mag:.9f} (want {want_mag:.9f})  "
            f"angle={res.angle:+.9f} (want {want_ang:+.9f})"
        )

    print()
    print("=== B2. Boundary cases: zero vector, most-negative integer, saturation ===")
    z = polar(0, 0)
    print(f"  zero vector (0,0)      -> mag={z.mag_q}, angle={z.angle_q}, "
          f"saturated={z.magnitude_saturated}")
    for label, (x, y) in (
        ("(INT32_MIN, 0)", (INT32_MIN, 0)),
        ("(INT32_MIN, INT32_MIN)", (INT32_MIN, INT32_MIN)),
        ("(INT32_MAX, 0)", (INT32_MAX, 0)),
    ):
        res = polar(x, y)
        print(f"  {label:22s} -> mag_q={res.mag_q}, saturated={res.magnitude_saturated}, "
              f"angle={res.angle:+.6f}")
    tiny = polar(2, 1)
    print(f"  tiny vector (2,1)      -> mag_q={tiny.mag_q} counts "
          f"(true sqrt(5)={math.sqrt(5):.4f} counts, rounds to 2), "
          f"angle_err={float(tiny.angle_error) * SCALE:+.2f} ulp")

    print()
    print("=== B3. Deterministic radius/angle enumeration (12 radii x 48 angles) ===")
    worst_a = worst_m = 0.0
    saturated = 0
    for ri in range(1, 13):
        rq = ri * (INT32_MAX // 13)
        for k in range(48):
            th = k * (2 * math.pi / 48)
            x = max(INT32_MIN, min(INT32_MAX, int(rq * math.cos(th))))
            y = max(INT32_MIN, min(INT32_MAX, int(rq * math.sin(th))))
            res = polar(x, y)
            if res.magnitude_saturated:
                saturated += 1
                continue
            worst_a = max(worst_a, abs(float(res.angle_error)) * SCALE)
            worst_m = max(worst_m, abs(float(res.mag_error)) * SCALE)
    print(f"  non-saturated points: max angle err = {worst_a:.2f} ulp, "
          f"max magnitude err = {worst_m:.2f} ulp; saturated points = {saturated}")
    print("  (analytic contract bound is 80 ulp; enumeration is a sample, not exhaustive)")

    print()
    print("=== B4. Vectoring-mode rejections (refused before any partial update) ===")
    good = polar(SCALE, SCALE)
    for label, fn in (
        ("x=True", lambda: polar(True, 0)),
        ("y='1'", lambda: polar(0, "1")),
        ("x=INT32_MAX+1", lambda: polar(INT32_MAX + 1, 0)),
        ("y=INT32_MIN-1", lambda: polar(0, INT32_MIN - 1)),
        ("polar_xy(inf,0)", lambda: polar_xy(math.inf, 0.0)),
        ("polar_xy(0,nan)", lambda: polar_xy(0.0, math.nan)),
    ):
        try:
            fn()
        except CordicError as exc:
            print(f"  {label:18s} -> {type(exc).__name__}")
    assert polar(SCALE, SCALE) == good  # no partial state leaks from failures
    print("  state after rejections is unchanged (pure function verified)")


def main() -> None:
    t0 = time.perf_counter()
    part_a()
    part_b()
    print()
    print(f"demo completed in {time.perf_counter() - t0:.2f}s (real computation, no sleep)")


if __name__ == "__main__":
    main()

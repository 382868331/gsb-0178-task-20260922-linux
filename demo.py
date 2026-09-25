"""Deterministic demo for the Q4.28 fixed-point CORDIC library.

Runs entirely offline with fixed inputs (no randomness, no network):

1. One normal sin/cos computation, printed with its per-round trace summary
   and its error against the independent Decimal reference.
2. A deterministic sweep over the full representable angle range reporting
   the maximum observed error (with an explicit non-exhaustive note).
3. One genuinely triggered rejection (out-of-range angle) and one triggered
   type rejection (bool), printed as caught exceptions.
4. The conservative error envelope (with_envelope=True): integer bounds for
   sin/cos of the quantized input angle, per-source error accounting, an
   independent containment check against the Decimal oracle, and envelope
   width statistics over a fixed grid.
"""

import time
from fractions import Fraction

from cordic_q428 import (
    HALF_PI_Q,
    INT32_MAX,
    INT32_MIN,
    SCALE,
    CordicError,
    float_to_q,
    sin_cos,
    sin_cos_radians,
    sin_cos_ref,
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
    print("=== 3. Triggered rejections (invalid input is refused, not corrected) ===")
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
    print("=== 4. Conservative error envelope (with_envelope=True) ===")
    q = float_to_q(1.0)
    r = sin_cos(q, with_envelope=True)
    env = r.envelope
    print(f"  angle_q = {q} (1.0 rad); envelope is for this exact quantized angle")
    print(f"  sin: output {r.sin_q}  in proven envelope [{env.sin_lo}, {env.sin_hi}]"
          f"  (width {env.sin_width} ulp)")
    print(f"  cos: output {r.cos_q}  in proven envelope [{env.cos_lo}, {env.cos_hi}]"
          f"  (width {env.cos_width} ulp)")
    src = env.sources
    print("  error sources (isolated envelope widths, ulp):")
    print(f"    gain quantization    : {float(src.gain_quantization_ulp):8.3f}")
    print(f"    atan-table quant.    : {float(src.table_quantization_ulp):8.3f}")
    print(f"    shift truncation     : {float(src.shift_truncation_ulp):8.3f}")
    print(f"    residual rotation    : {float(src.residual_rotation_ulp):8.3f}")
    print(f"  direction-branch unions at rounds: {list(env.union_rounds)}")
    # Independent containment check against the Decimal oracle.
    ref_sin, ref_cos = sin_cos_ref(q)
    ok = (env.sin_lo <= Fraction(ref_sin) * SCALE <= env.sin_hi
          and env.cos_lo <= Fraction(ref_cos) * SCALE <= env.cos_hi)
    print(f"  Decimal oracle inside envelope: {ok}; "
          f"output inside envelope: {env.contains(r.sin_q, r.cos_q)}")

    print()
    print("=== 5. Envelope width over a fixed grid (401 points) + boundary case ===")
    n = 401
    worst_w, worst_q = 0, 0
    t_sweep = time.perf_counter()
    for k in range(n):
        qq = INT32_MIN + k * (2**32 - 1) // (n - 1)
        e = sin_cos(qq, with_envelope=True).envelope
        w = max(e.sin_width, e.cos_width)
        if w > worst_w:
            worst_w, worst_q = w, qq
    print(f"  max envelope width = {worst_w} ulp at angle_q = {worst_q} "
          f"(grid of {n}, not exhaustive)")
    eb = sin_cos(HALF_PI_Q, with_envelope=True).envelope
    print(f"  quadrant boundary pi/2: boundary_union={eb.boundary_union} "
          f"(both legal reduction paths unioned), "
          f"sin envelope [{eb.sin_lo}, {eb.sin_hi}]")
    print(f"  envelope sweep took {time.perf_counter() - t_sweep:.2f}s")

    print()
    print(f"demo completed in {time.perf_counter() - t0:.2f}s (real computation, no sleep)")


if __name__ == "__main__":
    main()

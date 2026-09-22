"""定点 CORDIC（signed Q4.28，32 轮）演示：正常结果 + 实际触发的拒绝边界。

全部真实计算、确定性输入、离线运行；约数秒内完成。
"""

import random
import time
from decimal import Decimal

import cordic_q28 as cq


def show_normal_case():
    print("=== 1. 正常结果：angle = 1.0 rad（Q4.28 原始整数 268435456）===")
    ev = cq.evaluate(cq.SCALE)
    res = ev.result
    print(f"输入 angle_q      = {res.angle_q}")
    print(f"规约后 reduced_q  = {res.reduced_q}（象限符号 {res.quadrant_sign:+d}）")
    print(f"sin_q = {res.sin_q}  (~{Decimal(res.sin_q) / cq.SCALE:.10f})")
    print(f"cos_q = {res.cos_q}  (~{Decimal(res.cos_q) / cq.SCALE:.10f})")
    print(f"参考 sin = {ev.sin_ref:.12f}  误差 = {ev.sin_error_lsb:+.4f} ulp")
    print(f"参考 cos = {ev.cos_ref:.12f}  误差 = {ev.cos_error_lsb:+.4f} ulp")
    print("逐轮状态（前 3 轮与最后 1 轮，共 32 轮）：")
    for st in list(res.trace[:3]) + [res.trace[-1]]:
        print(
            f"  i={st.iteration:2d} d={st.direction:+d} "
            f"x={st.x:>11d} y={st.y:>11d} z={st.z:>11d}"
        )


def show_max_observed_error():
    print("\n=== 2. 采样核验：最大观测误差（相对 Decimal 高精度独立参考）===")
    rng = random.Random(20260922)  # 固定种子，确定性样本
    samples = [
        0,
        1,
        -1,
        cq.HALF_PI_Q,
        -cq.HALF_PI_Q,
        cq.PI_Q,
        -cq.PI_Q,
        cq.TWO_PI_Q,
        cq.INT32_MIN,
        cq.INT32_MAX,
    ]
    samples += [rng.randrange(cq.INT32_MIN, cq.INT32_MAX + 1) for _ in range(400)]
    worst = Decimal(0)
    worst_angle = 0
    for a in samples:
        ev = cq.evaluate(a)
        m = max(abs(ev.sin_error_lsb), abs(ev.cos_error_lsb))
        if m > worst:
            worst, worst_angle = m, a
    print(f"样本数            = {len(samples)}（含象限边界、最小角步、满范围端点）")
    print(f"最大观测误差      = {worst:.4f} ulp（1 ulp = 2**-28 ≈ 3.7e-9）")
    print(f"出现于 angle_q    = {worst_angle}（约 {Decimal(worst_angle) / cq.SCALE:.6f} rad）")
    print(
        "说明：以上为采样观测值，未穷举全部 2**32 个可表示输入；"
        "它是对所测样本的如实报告，不是数学上界证明。"
    )


def show_rejection():
    print("\n=== 3. 实际触发的拒绝边界 ===")
    for label, bad in [
        ("float NaN", float("nan")),
        ("bool True", True),
        ("超出 signed32 的整数", cq.INT32_MAX + 1),
    ]:
        try:
            cq.sin_cos(bad)
        except cq.CordicError as exc:
            print(f"  输入 {label!r:24} -> {type(exc).__name__}: {exc}")
        else:
            print(f"  输入 {label!r}: 未拒绝（不应发生）")


def main():
    t0 = time.perf_counter()
    show_normal_case()
    show_max_observed_error()
    show_rejection()
    print(f"\n总耗时 {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()

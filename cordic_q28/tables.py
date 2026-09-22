"""Q4.28 定点常量表：全部由 Decimal 高精度级数在导入时离线生成。

- pi 由 Machin 公式 pi = 16*atan(1/5) - 4*atan(1/239) 以 80 位精度计算；
- atan(2^-i) 表（i = 0..31）由泰勒级数 atan(x) = sum (-1)^k x^(2k+1)/(2k+1)
  生成（i = 0 时取 pi/4，避免 x = 1 处级数慢收敛）；
- 增益逆积 K_INV = prod_i 1/sqrt(1 + 2^-2i) 由 Decimal sqrt 高精度求积；
- 所有常量按 ties-to-even（ROUND_HALF_EVEN）量化到 Q4.28 整数。

生成过程只使用标准库 decimal，确定性、离线、可复现。
"""

from decimal import Decimal, localcontext, ROUND_HALF_EVEN

FRAC_BITS = 28
SCALE = 1 << FRAC_BITS  # 2**28，Q4.28 的比例因子
ITERATIONS = 32

# 表生成所用的高精度位数（十进制）
GEN_PRECISION = 80


def _atan_series(x: Decimal, prec: int) -> Decimal:
    """泰勒级数 atan(x)，要求 |x| < 1（x = 1 时请改用 pi/4）。"""
    with localcontext() as ctx:
        ctx.prec = prec
        x = +x
        x2 = x * x
        term = x
        total = term
        k = 1
        while True:
            term = term * x2
            add = term / (2 * k + 1)
            if k % 2 == 1:
                total -= add
            else:
                total += add
            if add == 0 or add.adjusted() < -(prec - 8):
                break
            k += 1
        return +total


def _compute_pi(prec: int = GEN_PRECISION) -> Decimal:
    """Machin 公式：pi = 16*atan(1/5) - 4*atan(1/239)。"""
    with localcontext() as ctx:
        ctx.prec = prec
        a = _atan_series(Decimal(1) / 5, prec + 8)
        b = _atan_series(Decimal(1) / 239, prec + 8)
        return +(16 * a - 4 * b)


# 高精度 pi（Decimal，80 位），同时供独立参考模块使用
PI = _compute_pi()


def quantize_q28(d: Decimal) -> int:
    """把高精度 Decimal 按 ties-to-even 量化成 Q4.28 原始整数。"""
    with localcontext() as ctx:
        ctx.prec = GEN_PRECISION
        return int((d * SCALE).to_integral_value(rounding=ROUND_HALF_EVEN))


def _build_atan_table() -> tuple[int, ...]:
    table = []
    for i in range(ITERATIONS):
        with localcontext() as ctx:
            ctx.prec = GEN_PRECISION
            if i == 0:
                a = +(PI / 4)  # atan(2**0) = pi/4
            else:
                a = _atan_series(Decimal(1) / (1 << i), GEN_PRECISION)
        table.append(quantize_q28(a))
    return tuple(table)


def _compute_gain_inverse() -> Decimal:
    """K_INV = prod_{i=0}^{31} 1/sqrt(1 + 2^-2i)（高精度）。"""
    with localcontext() as ctx:
        ctx.prec = GEN_PRECISION
        k = Decimal(1)
        for i in range(ITERATIONS):
            k *= (1 + Decimal(1) / (1 << (2 * i))).sqrt()
        return +(1 / k)


ATAN_TABLE: tuple[int, ...] = _build_atan_table()
"""atan(2^-i) 的 Q4.28 量化表，i = 0..31。"""

K_INV_Q: int = quantize_q28(_compute_gain_inverse())
"""32 级 CORDIC 增益逆积的 Q4.28 量化值（约 0.607252935）。"""

PI_Q: int = quantize_q28(PI)
"""pi 的 Q4.28 量化值。"""

TWO_PI_Q: int = quantize_q28(2 * PI)
"""2*pi 的 Q4.28 量化值，用于整数域角度规约。"""

HALF_PI_Q: int = quantize_q28(PI / 2)
"""pi/2 的 Q4.28 量化值（独立量化，未必恰为 PI_Q 的一半）。"""

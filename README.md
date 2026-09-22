# cordic_q28 — 定点 CORDIC 旋转的范围与误差核验

离线嵌入式数值工程用可复用模块：signed Q4.28 定点、32 轮旋转模式 CORDIC，
计算 `sin`/`cos`，输出逐轮状态并相对独立高精度参考给出误差。
纯 Python 标准库实现，核心为纯整数运算，不调用 `math.sin`/`cos`。

## 环境

- Python 3.14.7（Linux 原生，仅用标准库：`decimal`、`math`、`dataclasses` 等）
- 无需安装任何第三方依赖（`requirements.txt` 为空说明）
- 核心、测试、演示全部离线；不需要网络、账号、硬件或 Docker

## 接口

```python
import cordic_q28 as cq

res = cq.sin_cos(angle_q)        # 定点核心 -> CordicResult
ev  = cq.evaluate(angle_q)       # 核心结果 + 独立参考 + 误差 -> Evaluation
s, c = cq.reference_sin_cos(angle_q)  # 独立 Decimal 高精度参考（不调用核心算法）
r, sign = cq.reduce_angle(angle_q)    # 规约到 [-pi/2, pi/2] + 象限符号
```

- `angle_q`：signed Q4.28 原始整数（弧度），即真实角度 = `angle_q / 2**28`。
- `CordicResult`：`angle_q, reduced_q, quadrant_sign, sin_q, cos_q, trace`；
  `trace` 为 32 个 `RoundState(iteration, direction, x, y, z)`（每轮迭代后的状态）。
- `Evaluation`：`result, sin_ref, cos_ref, sin_error_lsb, cos_error_lsb`。
- 常量：`SCALE, FRAC_BITS, ITERATIONS, ATAN_TABLE, K_INV_Q, PI, PI_Q, TWO_PI_Q, HALF_PI_Q, INT32_MIN, INT32_MAX`。

## 输入上限与错误语义

| 输入 | 行为 |
|---|---|
| `int`，范围 `[-2**31, 2**31 - 1]`（约 [-16, 16) rad） | 正常计算 |
| `int` 超出 signed32 | `AngleRangeError`（也是 `ValueError`） |
| `bool` | `InvalidInputError`（整数参数拒绝 bool） |
| `float`（含 NaN、±Infinity，含整数值 float） | `InvalidInputError` |
| 其他类型（str、None、…） | `InvalidInputError`（也是 `TypeError`） |
| 迭代中间量/结果超出 signed32 | `CordicOverflowError`（也是 `OverflowError`） |

所有错误都是 `CordicError` 子类，消息含参数名或迭代轮次，可定位；
非法输入一律拒绝，不静默纠正。`sin_cos` 为纯函数，失败不留部分状态。

## 误差口径

- 误差单位 ulp = 2**-28 ≈ 3.73e-9：`error_lsb = sin_q - sin_ref * 2**28`（Decimal，带符号）。
- 参考值：`cordic_q28.reference` 用 80 位 pi（Machin 公式）做范围缩减、
  70 位 Decimal 泰勒级数独立求 sin/cos，参考自身误差远低于 1 ulp，可作真值。
  参考模块不调用被测核心算法；`sin²+cos²≈1` 的往返一致仅作补充检查。
- 观测水平：32 轮量化表 CORDIC 的固有误差为十几 ulp 量级
  （demo 在 410 个确定性样本上的最大观测误差约 12.5 ulp）。
  测试中的 32 ulp 界是带裕量的工程上界，非数学证明；demo 如实说明样本未穷举。

## 设计取舍

- **表生成**：`atan(2^-i)` 表与增益逆积 `K_INV = Π 1/√(1+2^-2i)` 在导入时由
  Decimal 高精度级数（atan 泰勒、Machin 公式求 pi、Decimal sqrt）生成，
  按 ties-to-even（ROUND_HALF_EVEN）量化为 Q4.28 整数；确定性、可复现。
  `atan(2^-i)`（i = 29..31）不足 1 ulp，量化为 0，属预期行为。
- **角度规约**：先在整数域按量化后的 `2π` 取模到 `[-π, π)`，再折叠到
  `[-π/2, π/2]` 并保留象限符号（`sin = sign·sin(r)`，`cos = sign·cos(r)`）。
- **右移**：一律算术右移、向负无穷取整（Python `>>` 语义），由 `_ashr` 显式固定。
- **溢出**：每轮迭代后的 x/y/z 及最终结果都做 signed32 检查，溢出即报错。
- **核心无浮点**：迭代路径只用整数加、减、移位；`decimal` 仅用于表生成与参考。

## 已知限制

- 只实现本题契约的旋转模式 sin/cos，不宣称完整 CORDIC 家族（向量模式、
  双曲、任意位宽）或任何行业标准兼容。
- 规约使用量化后的 `π`/`2π` 常量：大角度（|angle| 接近 16 rad）会引入
  至多几 ulp 的附加规约误差，这是定点设计的一部分，已计入上述观测误差。
- 算术右移的地板语义给 x/y 引入系统性小偏置（数 ulp），未做舍入补偿。
- 误差核验为采样观测（demo 410 个确定性样本），未穷举全部 2**32 个输入。
- `atan(2^-i)` 表末三项量化为 0，等效迭代步数略少于 32，已含在误差口径内。

## 命令

```bash
python -m unittest discover -s tests -v   # 24 个单元测试
python demo.py                            # 正常结果 + 最大观测误差 + 拒绝边界（约 1 秒内）
```

## 文件结构

```
cordic_q28/
  __init__.py    公开接口汇总
  errors.py      错误类型与语义
  tables.py      Decimal 高精度表生成（atan 表、增益逆积、pi 常量）
  core.py        定点 CORDIC 核心（纯整数，32 轮）
  reference.py   独立 Decimal 泰勒参考（不调用核心算法）
  analysis.py    evaluate：核心结果 vs 参考的误差
tests/test_cordic_q28.py
demo.py
```

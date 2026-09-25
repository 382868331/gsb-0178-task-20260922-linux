# cordic_q428 — 定点 CORDIC 旋转的范围与误差核验

离线嵌入式数值工程用可复用模块:signed Q4.28 定点、32 步旋转模式 CORDIC,
输出 sin/cos 整数、逐轮状态与相对高精度参考的误差;可选开关附带**保守误差包络**
(逐轮上下界传播证明)。纯 Python 标准库实现,无任何第三方依赖,完全离线运行。

## 环境

- Python 3.14.7(标准库即可,`requirements.txt` 为空约定,无需安装任何依赖)
- Linux 原生运行;不需要网络、账号、数据库、Docker 或硬件
- 下文命令均使用隔离环境解释器 `./.venv/bin/python`(venv 已按 Python 3.14.7 建好)

## 接口

```python
from cordic_q428 import sin_cos, sin_cos_radians, sin_cos_ref, compute_envelope

r = sin_cos(angle_q)                        # angle_q: int,Q4.28 弧度角,[-2**31, 2**31-1]
r = sin_cos(angle_q, with_envelope=True)    # 同上,并附带保守误差包络证明
r = sin_cos_radians(1.0)                    # 便捷封装:有限 float 弧度 -> Q4.28 -> sin_cos
r = sin_cos_radians(1.0, with_envelope=True)
s, c = sin_cos_ref(angle_q)                 # 独立 Decimal Taylor 参考(oracle),返回 Decimal
e = compute_envelope(angle_q)               # 只要包络证明、不要核心结果时用
```

`sin_cos` 返回 `CordicResult`(frozen dataclass):

| 字段 | 含义 |
|---|---|
| `sin_q`, `cos_q` | Q4.28 整数输出(sin/cos × 2^28) |
| `reduced_q` | 象限规约到 [-pi/2, pi/2] 后的角度 |
| `sign_flip` | 规约提取的象限符号(+1/-1) |
| `rounds` | 32 个 `RoundState(index, direction, x, y, z)` 逐轮状态 |
| `sin_error`, `cos_error` | Decimal:`输出/2^28 - 高精度参考`,带符号 |
| `envelope` | 默认 `None`;`with_envelope=True` 时为 `EnvelopeResult` 证明 |

常量:`SCALE=2**28`、`ATAN_TABLE_Q`(32 项)、`GAIN_INV_Q`、`PI_Q`、`HALF_PI_Q`。

## 保守误差包络(with_envelope 开关)

`with_envelope=True` **只增加证明信息**:旋转算法、向量递推、所有原有字段逐位不变
(有测试守护)。包络针对**明确的输入角量化值** `angle_q` 本身,不混入调用前未知的
角度误差;它用 `fractions.Fraction` 精确区间算术沿同一递推传播上下界,使最终区间
同时包含定点寄存器轨迹与该量化角的数学 sin/cos 真值。

`EnvelopeResult`(frozen dataclass)字段:

| 字段 | 含义 |
|---|---|
| `sin_lo`, `sin_hi`, `cos_lo`, `cos_hi` | sin/cos 的 **Q4.28 整数包络**(精确界向外取整) |
| `sin_lo_exact` 等 4 个 | 精确有理数(Fraction)上下界,单位 ulp |
| `sin_width_ulp`, `cos_width_ulp` | 包络宽度(ulp,Fraction) |
| `sources` | `ErrorSources`:三个误差来源各自的孤立半宽(ulp) |
| `rounds` | 32 个 `RoundEnvelope` 逐轮证明记录 |
| `reduced_q`, `sign_flip` | 与核心一致的规约结果 |
| `contains(s, c)` | 判定 Q4.28 量纲的值(int/Decimal/Fraction)是否在包络内 |

三类误差来源**分别计入**(各自单独传播一次,报告孤立贡献):

1. **增益量化**:`GAIN_INV_Q` 与真值的 ties-to-even 量化差 ≤ ½ ulp(另加 1e-60 ulp
   余量吸收表生成器 80 位 Decimal 求值误差),作为初始 x 区间。
2. **角度表量化**:每项 `atan(2^-i)` 表值 ±½ ulp 进入 z 更新;规约所用的量化 π
   同样按 `k × ½ ulp`(k 为减 π 次数)加宽规约后角度。
3. **算术右移截断**:sar 向负无穷取整,每次移位丢失 `[0, 1-2^-i)` ulp;移位按
   "整数网格上的 floor 移位 ∪ 精确除法"传播,逐轮记录实际截断间隙
   (`RoundEnvelope.shift_trunc`),测试逐轮验证其 ≤ `1 - 2^-i`。

分支边界取**并包络**:象限规约分段处理,z 区间跨 0 时两个合法方向路径各自独立
传播、仅在最终取并(中途按坐标盒合并会丢失角度相关性,故不做;路径数有 4096
防御上限,超限才退化为同号盒合并,仍然保守)。末轮残差角 z 按 sin/cos 的
1-Lipschitz 性折回 x/y 区间。

包络是保守的(实测宽度约 85–120 ulp),但**绝不退化为整个值域**(测试断言每个
输入宽度 < 4096 ulp,而值域为 2^31 ulp)。包含性由独立 Decimal 参考在零点、
各象限边界两侧 ±1 ulp 及一组固定角上核对,另有 101 点满量程扫描测试。

## 数值契约与错误语义

- **格式**:signed Q4.28,1 ulp = 2^-28 ≈ 3.73e-9,角度范围 [-8.0, 8.0) rad。
- **角度规约**:先加减量化 pi 规约到 [-pi/2, pi/2],记录象限符号,最后重新施加。
- **移位语义**:右移为算术移位、向负无穷取整(`sar`,即 Python `>>`)。
- **量化**:atan 表与增益逆乘积由 Decimal 高精度级数(80 位)生成,
  按 **ties-to-even**(ROUND_HALF_EVEN)量化到 Q4.28。
- **溢出**:任何 signed 32-bit 越界(输入或中间寄存器)抛
  `FixedPointOverflowError`,携带 `where`/`value` 定位信息,绝不回绕。
  包络传播同样如此:区间端点越界即抛 `FixedPointOverflowError`
  (`where="envelope.round[i]"`),**不用饱和掩盖**。
- **类型**:`bool` 不是合法定点整数,抛 `InvalidArgumentTypeError`;
  NaN/±Infinity 抛 `NonFiniteValueError`。非法输入一律拒绝,不静默纠正。
- 所有异常继承自 `CordicError`。`sin_cos` 是纯函数:调用失败不留任何部分状态。
- 核心与参考均**不调用** `math.sin/cos`(有测试守护)。

## 误差口径(浮点结果如何解读)

- `sin_error`/`cos_error` 是**带符号绝对误差**,单位是实数(弧度域的函数值),
  由 `输出/2^28` 减去 60 位精度 Decimal Taylor 参考得到;乘 `SCALE` 得 ulp 数。
- 参考本身经 Decimal 范围缩减 + Taylor 级数独立计算,**不调用被测核心**;
  参考对量化角(而非数学角)求值,故在 pi/2 等处残差 ~1 ulp 属预期。
- 解析误差上界取 **80 ulp**(增益量化 1 ulp + 32 轮移位截断经 ~1.647 增益
  放大 + 表量化),测试以此断言;全范围 2001 点扫描实测最大约 **13.7 ulp**
  (≈5.1e-8)。扫描是采样而非穷举,demo 中已注明。
- 包络(`with_envelope=True`)是**可证明的**包含界,与上述实测误差口径互补:
  它保证真值与定点输出都落在整数包络内,宽度约 85–120 ulp,比 80 ulp 的
  解析估计宽(区间传播天然保守),但对每个输入单独成立且经独立参考核对。
- 由于移位向负无穷取整,+q/-q 输出**不保证精确奇对称**,实测偏差 ≤ 7 ulp。

## 输入上限

- `angle_q`:signed 32-bit 整数,即 [-2147483648, 2147483647](弧度 [-8.0, 8.0))。
- `sin_cos_radians`:有限 float,|angle| < 8.0;8.0 本身不可表示,触发拒绝。
- 迭代数固定 32;规约循环预算 8 步(任意可表示角最多需 3 步),耗尽抛
  `RangeReductionError`(与溢出是不同异常类型)。包络路径数防御上限 4096。

## 设计取舍

- 规约用量化 pi 的整数加减而非浮点取模:全程定点、可复现、与硬件行为一致。
- 每次寄存器更新都做 32-bit 检查而非依赖 Python 大整数:精确定位溢出点。
- atan(1) 取 pi/4(Machin 公式)而非 x=1 的 Taylor 级数:后者收敛过慢。
- 误差以 Decimal 报告而非 float:避免在报告中再引入一次舍入。
- 包络用 Fraction 精确区间算术而非浮点区间:证明不受舍入影响,且逐轮可核验。
- 包络分支路径中途不合并:坐标盒合并会在 z≈0(如角度 k·π)处把瞬时分叉
  放大成值域级宽度;独立传播 + 末端取并保持包络紧凑且仍然保守。

## 已知限制

- 仅实现本题契约的旋转模式 sin/cos;不含向量模式、arctan、双曲 CORDIC,
  不宣称任何行业标准兼容。
- Q4.28 角度上限 ±8 rad;更大角度需调用方先自行规约(本库拒绝而非回绕)。
- demo 扫描为固定网格采样,非 2^32 穷举;穷举不在演示预算内。
- 包络的三个孤立来源宽度各自携带末轮残差角项,不必精确加和为总宽度;
  总包络(全源同开)才是正式包含界。

## 命令

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 回归入口:全部测试(60 项)
./.venv/bin/python demo.py                            # 约 4 秒:结果 + 扫描 + 拒绝 + 包络演示
```

## 目录结构

```
cordic_q428/
  __init__.py      公共 API 导出
  errors.py        异常层级(CordicError 及子类)
  fixedpoint.py    Q4.28 原语:sar、量化、32-bit 检查、输入验证
  tables.py        Decimal 高精度生成 atan 表 / 增益逆 / pi 常量
  reference.py     独立 Decimal Taylor 参考(oracle)
  cordic.py        32 步旋转模式 CORDIC 核心(with_envelope 开关)
  envelope.py      保守误差包络:Fraction 区间传播、来源归因、逐轮截断记录
tests/             unittest:契约、核心、参考独立性、包络包含性与截断界
demo.py            固定输入演示
```

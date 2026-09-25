# cordic_q428 — 定点 CORDIC 旋转的范围与误差核验

离线嵌入式数值工程用可复用模块:signed Q4.28 定点、32 步旋转模式 CORDIC,
输出 sin/cos 整数、逐轮状态与相对高精度参考的误差;可选开启**保守误差包络**,
围绕每轮定点运算传播上下界、分别计入各误差源,返回 sin/cos 的整数包络及误差
来源。纯 Python 标准库实现,无任何第三方依赖,完全离线运行。

## 环境

- Python 3.14.7(标准库即可,`requirements.txt` 为空约定,无需安装任何依赖)
- Linux 原生运行;不需要网络、账号、数据库、Docker 或硬件
- 下文命令中的 `./.venv/bin/python` 指仓库内隔离环境解释器(Python 3.14.7)

## 接口

```python
from cordic_q428 import sin_cos, sin_cos_with_envelope, sin_cos_radians, sin_cos_ref

r = sin_cos(angle_q)                        # angle_q: int,Q4.28 弧度角,范围 [-2**31, 2**31-1]
r = sin_cos(angle_q, with_envelope=True)    # 同上,并附加可核验的整数误差包络
r = sin_cos_with_envelope(angle_q)          # 上一行的命名便捷形式
r = sin_cos_radians(1.0)                    # 便捷封装:有限 float 弧度 -> Q4.28 -> sin_cos
s, c = sin_cos_ref(angle_q)                 # 独立 Decimal Taylor 参考(oracle),返回 Decimal
```

`sin_cos` 返回 `CordicResult`(frozen dataclass):

| 字段 | 含义 |
|---|---|
| `sin_q`, `cos_q` | Q4.28 整数输出(sin/cos × 2^28) |
| `reduced_q` | 象限规约到 [-pi/2, pi/2] 后的角度 |
| `sign_flip` | 规约提取的象限符号(+1/-1) |
| `rounds` | 32 个 `RoundState(index, direction, x, y, z)` 逐轮状态 |
| `sin_error`, `cos_error` | Decimal:`输出/2^28 - 高精度参考`,带符号 |
| `envelope` | 仅 `with_envelope=True` 时非 `None`:`SinCosEnvelope` 证明信息 |

`with_envelope` 开关**只增加证明信息**:旋转算法、逐轮状态与全部数值输出
与不开关时逐位一致(有测试守护),也不引入向量模式或任何新模式。

### 误差包络(`SinCosEnvelope`)

| 字段 | 含义 |
|---|---|
| `sin_lo/sin_hi`, `cos_lo/cos_hi` | 整数包络:真实 sin/cos(对**量化输入角** `angle_q/2^28`)乘 2^28 必落在闭区间内,核心整数输出同样被包含,故 `|输出-真值| ≤ hi-lo` |
| `sources` | `EnvelopeSources`:增益量化 / 角度表量化 / 移位截断三个误差源各自的包络宽度(其余源理想化单独传播),加残差旋转项,均为精确 `Fraction`(ulp) |
| `rounds` | 32 个 `RoundEnvelope(index, directions, x, y, z)` 逐轮区间轨迹 |
| `union_rounds` | z 区间跨 0、对 d=±1 两种合法旋转取并包络的轮次 |
| `boundary_union` | 输入恰落在 ±pi/2 规约边界、两条合法规约路径已取并包络 |
| `z_residual` | 32 轮后真实剩余角度的区间(ulp) |
| `contains(sin_q, cos_q)` | 输出是否落在包络内 |

包络语义要点:

- **针对明确的输入角量化值**:初态 z 是精确点,不混入任何调用前未知的角度误差。
- **三个误差源分别计入**:增益逆量化(精确值,≤0.5 ulp)、atan 表逐项量化
  (精确值,每项 ≤0.5 ulp,来自 80 位 Decimal 表 `ATAN_TABLE_DEC`/`GAIN_INV_DEC`)、
  算术右移截断(每轮每次移位损失 `f` ulp,`0 ≤ f ≤ 1-2^-i`,逐轮有测试验证)。
- **分支并包络**:象限规约分段处理,角度恰在 ±pi/2 时两条合法规约路径
  (停下,或再减/加一个 pi 并翻转象限符号)都求包络并取并;旋转内部 z 区间
  跨 0 时对 d=+1/-1 两个分支分别传播后取并。
- **残差旋转单独入账**:32 轮终止残留的角区间作为最后一次精确区间旋转施加,
  宽度单列 `residual_rotation_ulp`,不偷偷并入三个源。
- **不饱和、不掩盖**:包络用 `Fraction` 精确传播,越宽如实越宽,绝不截断到
  32 位或整个值域;真实数据通路溢出仍抛 `FixedPointOverflowError`。
- 全部区间运算为精确有理数,包络计算是纯函数,结果可复现。

常量:`SCALE=2**28`、`ATAN_TABLE_Q`(32 项)、`ATAN_TABLE_DEC`、`GAIN_INV_Q`、
`GAIN_INV_DEC`、`PI_Q`、`HALF_PI_Q`。

## 数值契约与错误语义

- **格式**:signed Q4.28,1 ulp = 2^-28 ≈ 3.73e-9,角度范围 [-8.0, 8.0) rad。
- **角度规约**:先加减量化 pi 规约到 [-pi/2, pi/2],记录象限符号,最后重新施加。
- **移位语义**:右移为算术移位、向负无穷取整(`sar`,即 Python `>>`)。
- **量化**:atan 表与增益逆乘积由 Decimal 高精度级数(80 位)生成,
  按 **ties-to-even**(ROUND_HALF_EVEN)量化到 Q4.28。
- **溢出**:任何 signed 32-bit 越界(输入或中间寄存器)抛
  `FixedPointOverflowError`,携带 `where`/`value` 定位信息,绝不回绕。
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
- `envelope`(开启 `with_envelope` 时)是**证明出来的包含界**,与上面的
  事后观测互补:`[sin_lo, sin_hi]` 同时包含真实值与整数输出,宽度即误差上界。
  典型角(如 1.0 rad)包络宽约 40–50 ulp;包络保守、允许偏宽,但对任何输入
  都不是"整个值域"式的平凡答案(demo 第 5 节给出网格上的最大宽度实测)。
- 包络的独立核对:测试用 Decimal 高精度级数在零、每个象限边界两侧 ±1 ulp
  和一组固定角上验证参考值落在包络内,并逐轮验证移位截断界
  `0 ≤ f ≤ 1-2^-i` 与寄存器始终落在传播区间内。
- 由于移位向负无穷取整,+q/-q 输出**不保证精确奇对称**,实测偏差 ≤ 7 ulp。

## 输入上限

- `angle_q`:signed 32-bit 整数,即 [-2147483648, 2147483647](弧度 [-8.0, 8.0))。
- `sin_cos_radians`:有限 float,|angle| < 8.0;8.0 本身不可表示,触发拒绝。
- 迭代数固定 32;规约循环预算 8 步(任意可表示角最多需 3 步),耗尽抛
  `RangeReductionError`(与溢出是不同异常类型)。

## 设计取舍

- 规约用量化 pi 的整数加减而非浮点取模:全程定点、可复现、与硬件行为一致。
- 每次寄存器更新都做 32-bit 检查而非依赖 Python 大整数:精确定位溢出点。
- atan(1) 取 pi/4(Machin 公式)而非 x=1 的 Taylor 级数:后者收敛过慢。
- 误差以 Decimal 报告而非 float:避免在报告中再引入一次舍入。
- 包络引擎用 `Fraction` 精确区间运算而非浮点区间:证明不受二次舍入影响;
  分支并包络让包络不依赖具体走哪条路径,对规约边界与 z 符号判决都稳健。
- 包络开关做成 `sin_cos` 的关键字参数而非新算法:证明信息与数值计算解耦,
  原路径逐位不变。

## 已知限制

- 仅实现本题契约的旋转模式 sin/cos;不含向量模式、arctan、双曲 CORDIC,
  不宣称任何行业标准兼容。
- Q4.28 角度上限 ±8 rad;更大角度需调用方先自行规约(本库拒绝而非回绕)。
- demo 扫描为固定网格采样,非 2^32 穷举;穷举不在演示预算内。
- 包络是保守证明:对 z 区间恰好在早期轮次跨 0 的个别输入,并包络会显著
  变宽(实测网格最大约 7.1e3 ulp,仍远小于值域 2^32);这是证明的固有保守性,
  不是饱和或截断。

## 命令

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 回归入口:全部 60 项测试
./.venv/bin/python demo.py                            # 约 2.5 秒:结果 + 扫描 + 拒绝 + 包络演示
```

## 目录结构

```
cordic_q428/
  __init__.py      公共 API 导出
  errors.py        异常层级(CordicError 及子类)
  fixedpoint.py    Q4.28 原语:sar、量化、32-bit 检查、输入验证
  tables.py        Decimal 高精度生成 atan 表 / 增益逆 / pi 常量(含量化前 Decimal 表)
  reference.py     独立 Decimal Taylor 参考(oracle)
  cordic.py        32 步旋转模式 CORDIC 核心(含 with_envelope 开关)
  envelope.py      保守误差包络:逐轮区间传播、分支并包络、误差源分列
tests/             unittest:契约、核心、参考独立性、包络包含性与逐轮截断界
demo.py            固定输入演示(含包络与边界并包络展示)
```

# cordic_q428 — 定点 CORDIC 旋转与向量模式(atan2/模长)的范围与误差核验

离线嵌入式数值工程用可复用模块:signed Q4.28 定点、32 步 CORDIC,提供两种模式——
旋转模式输出 sin/cos,向量模式一次整数移位迭代同时输出 **atan2 角度与模长**;
均给出整数结果、逐轮状态与相对高精度独立参考的误差。纯 Python 标准库实现,
无任何第三方依赖,完全离线运行。

## 环境

- Python 3.14.7(标准库即可,`requirements.txt` 为空约定,无需安装任何依赖)
- Linux 原生运行;不需要网络、账号、数据库、Docker 或硬件

## 接口

```python
from cordic_q428 import sin_cos, sin_cos_radians, sin_cos_ref
from cordic_q428 import polar, polar_xy, polar_ref

r = sin_cos(angle_q)          # angle_q: int,Q4.28 弧度角,范围 [-2**31, 2**31-1]
r = sin_cos_radians(1.0)      # 便捷封装:有限 float 弧度 -> Q4.28 -> sin_cos
s, c = sin_cos_ref(angle_q)   # 独立 Decimal Taylor 参考(oracle),返回 Decimal

p = polar(x_q, y_q)           # 向量模式:x,y 为 Q4.28 整数 -> 模长 + atan2(y,x)
p = polar_xy(3.0, 4.0)        # 便捷封装:两个有限 float 坐标 -> Q4.28 -> polar
a, m = polar_ref(x_q, y_q)    # 独立 Decimal 参考:atan 级数 + Decimal sqrt
```

`sin_cos` 返回 `CordicResult`(frozen dataclass):

| 字段 | 含义 |
|---|---|
| `sin_q`, `cos_q` | Q4.28 整数输出(sin/cos × 2^28) |
| `reduced_q` | 象限规约到 [-pi/2, pi/2] 后的角度 |
| `sign_flip` | 规约提取的象限符号(+1/-1) |
| `rounds` | 32 个 `RoundState(index, direction, x, y, z)` 逐轮状态 |
| `sin_error`, `cos_error` | Decimal:`输出/2^28 - 高精度参考`,带符号 |

`polar` 返回 `PolarResult`(frozen dataclass):

| 字段 | 含义 |
|---|---|
| `mag_q` | 模长 `hypot(x,y)` 的 Q4.28 整数;真实模长超过有符号 32 位上限时钳到 `INT32_MAX` |
| `angle_q` | `atan2(y,x)` 的 Q4.28 整数,范围 `[-pi, pi]`(零向量约定为 0) |
| `angle_offset_q` | 象限折叠后加回的角度偏移:0 / ±pi/2 / ±pi |
| `magnitude_saturated` | 真实模长 `x²+y² > INT32_MAX²` 时为 `True`(精确整数谓词判定) |
| `rounds` | 32 个折叠后向量的逐轮状态;`direction` 为本轮微旋转方向(+1/-1),对齐后为 0 |
| `mag_error`, `angle_error` | Decimal:`输出/2^28 - 独立 Decimal 参考`,带符号 |

另有只读视图 `r.sin/r.cos` 与 `p.mag/p.angle`(float)。

常量:`SCALE=2**28`、`ATAN_TABLE_Q`(32 项)、`GAIN_INV_Q`、
`GAIN_PREFIX_INV_Q`(33 项逐轮前缀增益逆,向量模式按实际旋转轮数补偿)、
`PI_Q`、`HALF_PI_Q`。

## 数值契约与错误语义

- **格式**:signed Q4.28,1 ulp = 2^-28 ≈ 3.73e-9,坐标与角度范围 [-8.0, 8.0)。
- **旋转模式象限规约**:先加减量化 pi 规约到 [-pi/2, pi/2],记录象限符号,最后重新施加。
- **向量模式象限折叠**:把绝对值较大的分量折到非负 x 轴(主导轴折叠),折叠角恒在
  [-pi/4, pi/4],四个坐标轴都直接折到 +x、零次微旋转,故轴上结果逐位精确;
  记录 `angle_offset_q`(0/±pi/2/±pi)在末尾加回。
- **向量模式归一化**:折叠后将两分量同额左移(纯整数移位,角度不变)到接近满量程再
  迭代,使极小向量也保留角度精度;模长在增益补偿后按 ties-to-even 右移还原。
- **移位语义**:右移为算术移位、向负无穷取整(`sar`,即 Python `>>`)。
- **量化**:atan 表、增益逆乘积及前缀增益由 Decimal 高精度级数(80 位)生成,
  按 **ties-to-even**(ROUND_HALF_EVEN)量化到 Q4.28。
- **增益**:每轮微旋转把模长放大 `sqrt(1+2^-2i)`;向量模式在 y 精确归零后停止旋转,
  并仅按**实际执行的前缀轮数**用 `GAIN_PREFIX_INV_Q[m]` 补偿。
- **溢出(旋转模式)** 与 **向量模式 x/y/z 中间寄存器**:任何越界抛
  `FixedPointOverflowError`,携带 `where`/`value`,绝不回绕。向量模式 x/y 的合法
  工作范围是一个解析的"宽包络"(满量程 × √2 × 前向增益 1.65 + 余量),仍受检查。
- **模长饱和(向量模式唯一不抛错的输出越界)**:真实模长 > INT32_MAX(即 8.0 − 1ulp)
  时,`mag_q` 钳到 `INT32_MAX` 且 `magnitude_saturated=True`;此时 `mag_error` 如实
  反映饱和缺口(为负)。角度在饱和时仍然有效。包络内因截断噪声(解析上界
  `_MAG_NOISE_CLIP=128` 计数)略微越界则静默钳到 `INT32_MAX` 但**不**置饱和标志;
  超过该噪声预算仍抛错。
- **零向量**:`polar(0,0)` 返回模长 0、角度 0、不旋转、不饱和(明确结果,非异常)。
- **负 x 轴约定**:整数没有"有符号零",`atan2(0, x<0)` 定义为 **+pi**。
- **类型**:`bool` 不是合法定点整数,抛 `InvalidArgumentTypeError`;
  NaN/±Infinity 抛 `NonFiniteValueError`。非法输入一律在任何迭代前拒绝,不静默纠正。
- 所有异常继承自 `CordicError`。`sin_cos`/`polar` 均为纯函数:调用失败不留任何
  部分状态(有测试守护)。
- 核心与参考均**不调用** `math.sin/cos/tan/atan2/hypot`(有测试守护)。

## 误差口径(浮点结果如何解读)

- `*_error` 是**带符号绝对误差**,单位是实数(角度为弧度、模长为 Q4.28 实数刻度),
  由 `输出/2^28` 减去 60 位精度 Decimal 独立参考得到;乘 `SCALE` 得 ulp 数。
- 参考与被测核心**无共享代码路径**:`sin_cos_ref` 用 Decimal 范围缩减 + Taylor;
  `polar_ref` 用符号分象限 + Decimal atan 级数 + Decimal sqrt,且对**量化输入**求值。
- 解析误差上界取 **80 ulp**(每轮移位截断 ≤1 计数经 ~1.647 增益放大 + 表/增益量化),
  测试对非饱和点以此断言。
  - 旋转模式:全范围 801 点固定网格实测最大约 **12.4 ulp**。
  - 向量模式:40000 点确定性哈希枚举(8533 点约 21% 因真实模长超上限被饱和跳过)
    实测角度最大 **约 4.2 ulp**、模长最大 **约 17.9 ulp**;近轴窄角带(y=±1)
    角度 ≤ **2 ulp**。枚举是采样而非穷举,demo 中已注明。
- 由于移位向负无穷取整,+q/-q 输出**不保证精确奇对称**。
- 模长在 Q4.28 下最小分辨为 1 计数:如 `(2,1)` 真实模长 √5≈2.236 计数,
  ties-to-even 量化为 2,属预期;**角度**经归一化不受此低位分辨率影响。

## 输入上限

- 旋转模式 `angle_q`:signed 32-bit 整数 [-2147483648, 2147483647](弧度 [-8.0, 8.0))。
- 向量模式 `x_q`,`y_q`:各自为 signed 32-bit Q4.28 整数;模长上限为
  `INT32_MAX`(约 7.999999996),超过即按上面规则饱和。
- `sin_cos_radians` / `polar_xy`:有限 float,绝对值 < 8.0;越界拒绝。
- 迭代数固定 32;旋转模式规约循环预算 8 步(任意可表示角最多需 3 步),耗尽抛
  `RangeReductionError`(与溢出是不同异常类型)。

## 设计取舍

- 向量模式用**主导轴折叠**而非仅折到 x≥0:折叠角更小、四轴逐位精确;配合
  **预移位归一化**,小向量角度不再被早期 `x>>i` 截断破坏(全程仍是整数移位迭代)。
- 模长增益按**实际旋转的前缀轮数**补偿(`GAIN_PREFIX_INV_Q`),使精确角逐位收敛,
  而不是强制对已对齐向量继续旋转再收敛。
- 饱和用精确整数谓词 `x²+y² > INT32_MAX²` 判定,不依赖补偿结果的舍入。
- 规约/折叠用整数加减而非浮点取模:全程定点、可复现、与硬件行为一致。
- atan(1) 取 pi/4(Machin 公式)而非 x=1 的 Taylor 级数:后者收敛过慢。
- 误差以 Decimal 报告而非 float:避免在报告中再引入一次舍入。

## 已知限制

- 仅实现旋转模式 sin/cos 与向量模式 atan2/模长;不含双曲 CORDIC 等,
  不宣称任何行业标准兼容。
- 字长固定 Q4.28、迭代固定 32;不支持任意字长,不提供逐步误差区间证书或提前停止认证。
- 角度/坐标上限 ±8 rad/单位;向量模长上限 ~8.0,超出按饱和处理(本库钳位而非回绕)。
- `atan2(0, 负x)` 定义为 +pi(无有符号整数零)。
- demo 扫描/枚举为固定采样,非 2³² 穷举;穷举不在演示预算内。

## 命令

```bash
python -m unittest discover -s tests -v   # 运行全部测试(83 项)
python demo.py                            # 约 1 秒内完成:两模式正常结果 + 边界 + 拒绝演示
```

## 目录结构

```
cordic_q428/
  __init__.py      公共 API 导出
  errors.py        异常层级(CordicError 及子类)
  fixedpoint.py    Q4.28 原语:sar、量化、32-bit 检查、输入验证、定点乘法
  tables.py        Decimal 高精度生成 atan 表 / 增益逆 / 前缀增益 / pi 常量
  reference.py     独立 Decimal 参考:sin/cos Taylor、atan2/sqrt oracle
  cordic.py        32 步旋转模式与向量模式 CORDIC 核心
tests/             unittest:契约、核心、参考独立性、向量模式(atan2/模长/饱和/非法输入)
demo.py            固定输入演示(两模式)
```

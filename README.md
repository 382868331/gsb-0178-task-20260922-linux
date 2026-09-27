# cordic_q428 — 定点 CORDIC 旋转的范围与误差核验

离线嵌入式数值工程用可复用模块:signed Q4.28 定点、32 步旋转模式 CORDIC,
输出 sin/cos 整数、逐轮状态与相对高精度参考的误差。纯 Python 标准库实现,
无任何第三方依赖,完全离线运行。

## 环境

- Python 3.14.7(标准库即可,`requirements.txt` 为空约定,无需安装任何依赖)
- Linux 原生运行;不需要网络、账号、数据库、Docker 或硬件

## 接口

```python
from cordic_q428 import sin_cos, sin_cos_radians, sin_cos_ref

r = sin_cos(angle_q)          # angle_q: int,Q4.28 弧度角,范围 [-2**31, 2**31-1]
r = sin_cos(angle_q, max_error=1e-6)  # 给出绝对误差容差,允许提前结束
r = sin_cos_radians(1.0)      # 便捷封装:有限 float 弧度 -> Q4.28 -> sin_cos
s, c = sin_cos_ref(angle_q)   # 独立 Decimal Taylor 参考(oracle),返回 Decimal
```

`sin_cos` 返回 `CordicResult`(frozen dataclass):

| 字段 | 含义 |
|---|---|
| `sin_q`, `cos_q` | Q4.28 整数输出(sin/cos × 2^28) |
| `reduced_q` | 象限规约到 [-pi/2, pi/2] 后的角度 |
| `sign_flip` | 规约提取的象限符号(+1/-1) |
| `rounds` | 32 个 `RoundState(index, direction, x, y, z)` 逐轮状态 |
| `sin_error`, `cos_error` | Decimal:`输出/2^28 - 高精度参考`,带符号 |
| `steps_executed` | 实际执行的轮数;默认 32,提前结束时小于 32 |
| `error_bound` | Decimal,对**本次输出**成立的绝对误差解析上界(实数单位) |
| `requested_tolerance` | 调用方给的 `max_error`,默认完整计算时为 `None` |
| `tolerance_met` | 保证是否满足:有容差时为 `True`(否则抛异常),完整计算时为 `None` |

**默认用法不变**:`sin_cos(angle_q)` 不带 `max_error` 时仍执行全部 32 轮。
`max_error` 为仅限关键字参数,接受正的 `int`/`float`/`Decimal`(实数单位的
绝对误差,与 `sin_error` 同口径);`bool`、NaN/Inf、零与负数一律以
`InvalidArgumentTypeError` 拒绝。提前结束时 `rounds` 长度等于 `steps_executed`。

常量:`SCALE=2**28`、`ATAN_TABLE_Q`(32 项)、`GAIN_INV_Q`、`PI_Q`、
`HALF_PI_Q`,以及 `ERROR_BOUNDS_REAL`(每轮一条解析界)、
`MIN_GUARANTEED_ERROR_REAL`/`MIN_GUARANTEED_ERROR_STEPS`、
`error_bound_real(n)`、`error_bound_ulp(n)`。

## 提前结束与误差保证

提前结束**绝不**依据“相邻两轮变化很小”——步间变化看不到尚未旋转掉的残角,
也不含定点截断。程序在每轮后比对一条**先验最坏误差界**(80 位 Decimal
预计算,见 `cordic_q428/bounds.py`),首次满足 `B_n <= max_error` 即停止。
对量化输入角 `angle_q/2^28` 的精确 sin/cos,每个输出坐标的绝对误差都不超过
`B_n`(Q4.28 ulp),由四部分组成:

1. **残角**:`n` 轮贪心微旋转后残角满足标准 CORDIC 残角定理
   `|theta_n| <= atan(2^{-(n-1)})`;表项 ties-to-even 量化的 `n/2` ulp
   累计余量同时覆盖符号翻转点附近偏离理想轨迹的情形。数据路径以 32 步
   逆增益初始化,`n` 轮后范数 ≤ 1,残角对任一坐标的扰动上界即残角本身。
2. **增益失配**:提前停止留下尾部增益
   `G_n = prod_{j=n}^{31} sqrt(1+2^{-2j})` 未抵消,相对幅值误差 ≤ `G_n-1`。
3. **移位截断**:每轮算术右移注入欧氏范数 ≤ `sqrt(2)` ulp 的误差向量,
   后续微旋转使范数精确乘上 `f_j = sqrt(1+2^{-2j})`;累计因子满足
   `A_0=0`、`A_n = 1 + f_{n-1} A_{n-1}`。
4. **初始增益量化**:`GAIN_INV_Q` 的 0.5 ulp 经 `n` 轮放大 ≤ `0.5*K_n`。

合计:

`B_n = (K_n/K_32)(atan(2^{-(n-1)})*2^28 + n/2) + (G_n-1)*2^28 + sqrt(2) A_n + 0.5 K_n`

该界与角度无关、不依赖采样;32 轮时约 **63 ulp**,与既有 80 ulp 的宽松解析
上界一致;全格式最小可保证误差约 **56.7 ulp**(≈2.11e-7,出现在第 28 轮,
之后残角表量化使界重新增大)。

**无法保证时明确失败**:若 `max_error < MIN_GUARANTEED_ERROR_REAL`,即使
跑满 32 轮也无法证明满足,立即抛出 `ToleranceNotGuaranteedError`(继承
`CordicError`),携带 `requested_tolerance`、可达到的 `best_bound` 与对应
`steps`;库不会悄悄放宽容差,也不会返回未获保证的结果。

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
- 由于移位向负无穷取整,+q/-q 输出**不保证精确奇对称**,实测偏差 ≤ 7 ulp。

## 输入上限

- `angle_q`:signed 32-bit 整数,即 [-2147483648, 2147483647](弧度 [-8.0, 8.0))。
- `sin_cos_radians`:有限 float,|angle| < 8.0;8.0 本身不可表示,触发拒绝。
- 默认迭代数 32;给出 `max_error` 时在解析误差界首次达标时提前停止,
  无法保证时抛 `ToleranceNotGuaranteedError`。规约循环预算 8 步(任意
  可表示角最多需 3 步),耗尽抛 `RangeReductionError`(与溢出是不同异常类型)。

## 设计取舍

- 规约用量化 pi 的整数加减而非浮点取模:全程定点、可复现、与硬件行为一致。
- 每次寄存器更新都做 32-bit 检查而非依赖 Python 大整数:精确定位溢出点。
- atan(1) 取 pi/4(Machin 公式)而非 x=1 的 Taylor 级数:后者收敛过慢。
- 误差以 Decimal 报告而非 float:避免在报告中再引入一次舍入。

## 已知限制

- 仅实现本题契约的旋转模式 sin/cos;不含向量模式、arctan、双曲 CORDIC,
  不宣称任何行业标准兼容。
- Q4.28 角度上限 ±8 rad;更大角度需调用方先自行规约(本库拒绝而非回绕)。
- demo 扫描为固定网格采样,非 2^32 穷举;穷举不在 8 秒演示预算内。

## 命令

```bash
python -m unittest discover -s tests -v   # 运行全部测试(57 项)
python demo.py                            # 约 1 秒内完成:正常结果 + 扫描 + 拒绝演示
```

## 目录结构

```
cordic_q428/
  __init__.py      公共 API 导出
  errors.py        异常层级(CordicError 及子类)
  fixedpoint.py    Q4.28 原语:sar、量化、32-bit 检查、输入验证
  tables.py        Decimal 高精度生成 atan 表 / 增益逆 / pi 常量
  bounds.py        提前停止的每步解析误差上界(残角/截断/量化)
  reference.py     独立 Decimal Taylor 参考(oracle)
  cordic.py        32 步旋转模式 CORDIC 核心 + 容差提前停止
tests/             unittest:契约、核心、参考独立性、提前停止误差保证
demo.py            固定输入演示
```

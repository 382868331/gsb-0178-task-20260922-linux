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
                             # 默认固定 32 步(原始完整计算)
r = sin_cos(angle_q, tol_ulp=300)          # 允许提前结束:误差以 ulp 计
r = sin_cos(angle_q, tolerance=1e-7)       # 或:绝对误差按实数单位
r = sin_cos_radians(1.0)      # 便捷封装:有限 float 弧度 -> Q4.28 -> sin_cos
                             # 同样接受 tolerance=/tol_ulp= 关键字
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

容差模式新增字段(默认模式下也会给出 `error_bound_ulp`,但 `tolerance_met` 为 `None`):

| 字段 | 含义 |
|---|---|
| `steps` | 实际执行的轮数(1..32);提前结束时小于 32 |
| `early_stopped` | 是否在 32 轮之前停止 |
| `error_bound_ulp` | 可证的每坐标绝对误差上界(精确 `Fraction`,单位 ulp);`error_bound` 属性为实数单位 |
| `tolerance` | 接受的容差(实数单位 `Fraction`),未给时为 `None` |
| `tolerance_met` | `True`/`False`/`None`(未给容差时);失败时调用直接抛异常 |

判定规则:每轮结束后用**可证上界**而非相邻两轮输出变化判断。若全部 32 轮(含所有中间轮)
的最优上界仍不满足容差,抛 `ToleranceNotGuaranteedError`(`steps=32`,异常携带
`bound` 与完整 32 步 `result`),绝不悄悄放宽。`tolerance` 与 `tol_ulp` 互斥;
接受 `int`/`float`/`Decimal`/`Fraction`(`tolerance`)或正整数(`tol_ulp`);
bool、NaN/Infinity、非正值一律拒绝(`InvalidArgumentTypeError`/
`InvalidArgumentValueError`;其中 NaN/Infinity 抛 `NonFiniteValueError`)。

常量:`SCALE=2**28`、`ATAN_TABLE_Q`(32 项)、`GAIN_INV_Q`、`PI_Q`、`HALF_PI_Q`。

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

### 提前终止的可证误差上界

容差模式的判定基于每轮 n 的严格上界,全程用精确 `Fraction` 运算(无浮点舍入),
组成(均为每坐标、单位 ulp,S=2^28):

- **角度残差**:`|z_n| + n/2`。量化表项与真 atan 之差 ≤ 1/2 ulp,故真角度的理想残差
  以 `|z_n| + n/2` 为界;未旋转的角度直接映射为同量级的 sin/cos 误差。
- **移位截断**:每轮两个坐标各有 (-1, 0] ulp 截断(范数 ≤ √2 ulp),经后续轮次
  以 CORDIC 增益 K_32 ≈ 1.647 放大,取严格有理上界 17/12 · K_32 · n
  (`√2 < 17/12`,`sqrt(1+t) < 1+t/2` 给出 K 的精确乘积上界)。
- **增益逆初值量化**:`x0` 与真 1/K_32 之差 ≤ 1/2 ulp,经 n 轮旋转以 K_32 放大。
- **未补偿增益**:n 轮只施加 K_n,种子却含 1/K_32;相对幅度亏损
  `K_32/K_n - 1 ≤ ∏_{i=n}^{31}(1+2^{-2i-1}) - 1`,作用在单位向量上。
- **象限规约**:每次量化 pi 加减与真 pi 之差 ≤ 1/2 ulp,任意 Q4.28 角规约最多
  3 次,合计 ≤ 3/2 ulp,按角度误差计入。

因此**不会**因为相邻两轮变化很小(表项在 i≥29 时量化为 0,z 完全不变)就宣称收敛。
上界随 n 非单调(z 可能越过 0),失败时报告所有已到达轮次中的**最优**上界;
32 步时该上界约 **98.1 ulp**,中间轮最优约 **80.2 ulp**(即当前 Q4.28 表示可保证的
下限附近)。实测误差仍远小于该保守上界(2001 点扫描最大 13.7 ulp,且有测试断言
每一点的实际误差不超过当次证明的上界)。

## 输入上限

- `angle_q`:signed 32-bit 整数,即 [-2147483648, 2147483647](弧度 [-8.0, 8.0))。
- `sin_cos_radians`:有限 float,|angle| < 8.0;8.0 本身不可表示,触发拒绝。
- 默认迭代数固定 32(不传容差时行为不变);给出容差时可在 1..32 轮间提前结束,
  无法保证时抛 `ToleranceNotGuaranteedError`。规约循环预算 8 步(任意可表示角最多需 3 步),耗尽抛
  `RangeReductionError`(与溢出是不同异常类型)。

## 设计取舍

- 规约用量化 pi 的整数加减而非浮点取模:全程定点、可复现、与硬件行为一致。
- 每次寄存器更新都做 32-bit 检查而非依赖 Python 大整数:精确定位溢出点。
- atan(1) 取 pi/4(Machin 公式)而非 x=1 的 Taylor 级数:后者收敛过慢。
- 误差以 Decimal 报告而非 float:避免在报告中再引入一次舍入。

## 已知限制

- 仅实现本题契约的旋转模式 sin/cos;不含向量模式、arctan、双曲 CORDIC,
  不宣称任何行业标准兼容。
- Q4.28 角度上限 ±8 rad;更大角度需调用方先自行规约(本库拒绝而非回绕)。
- 可保证误差受 Q4.28 舍入限制:全部 32 轮的最优可证上界约 80.2 ulp;更严容差
  会被明确拒绝而非放宽。该上界是保守数学界,实测误差约小一个数量级。
- demo 扫描为固定网格采样,非 2^32 穷举;穷举不在 8 秒演示预算内。

## 命令

```bash
python -m unittest discover -s tests -v   # 运行全部测试(54 项)
python demo.py                            # 约 4 秒内完成:正常结果 + 扫描 + 提前终止/拒绝演示
```

## 目录结构

```
cordic_q428/
  __init__.py      公共 API 导出
  errors.py        异常层级(CordicError 及子类)
  fixedpoint.py    Q4.28 原语:sar、量化、32-bit 检查、输入验证
  tables.py        Decimal 高精度生成 atan 表 / 增益逆 / pi 常量
  reference.py     独立 Decimal Taylor 参考(oracle)
  cordic.py        32 步旋转模式 CORDIC 核心
tests/             unittest:契约、核心、参考独立性
demo.py            固定输入演示
```

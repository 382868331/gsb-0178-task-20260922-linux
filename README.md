# cordic_q428 — 定点 CORDIC 旋转的范围与误差核验

离线嵌入式数值工程用可复用模块:signed Q4.28 定点、32 步旋转模式 CORDIC,
输出 sin/cos 整数、逐轮状态与相对高精度参考的误差;另提供双曲旋转入口,
输出 cosh/sinh。纯 Python 标准库实现,无任何第三方依赖,完全离线运行。

## 环境

- Python 3.14.7(标准库即可,`requirements.txt` 为空约定,无需安装任何依赖)
- Linux 原生运行;不需要网络、账号、数据库、Docker 或硬件

## 隔离 venv 准备

```bash
python3.14 -m venv .venv        # 隔离环境,避免污染系统 Python
. .venv/bin/activate
pip install -r requirements.txt  # 空约定:不安装任何第三方包
```

不建 venv 直接用系统 Python 3.14.7 也可以,库只依赖标准库。

## 接口

```python
from cordic_q428 import sin_cos, sin_cos_radians, sin_cos_ref

r = sin_cos(angle_q)          # angle_q: int,Q4.28 弧度角,范围 [-2**31, 2**31-1]
r = sin_cos_radians(1.0)      # 便捷封装:有限 float 弧度 -> Q4.28 -> sin_cos
s, c = sin_cos_ref(angle_q)   # 独立 Decimal Taylor 参考(oracle),返回 Decimal

from cordic_q428 import sinh_cosh, sinh_cosh_ref

h = sinh_cosh(z_q)              # z_q: int,Q4.28 双曲角,只验收 |z| <= 1(即 |z_q| <= SCALE)
s, c = sinh_cosh_ref(z_q)       # 独立 Decimal 指数定义参考(oracle),返回 Decimal
```

`sin_cos` 返回 `CordicResult`(frozen dataclass):

| 字段 | 含义 |
|---|---|
| `sin_q`, `cos_q` | Q4.28 整数输出(sin/cos × 2^28) |
| `reduced_q` | 象限规约到 [-pi/2, pi/2] 后的角度 |
| `sign_flip` | 规约提取的象限符号(+1/-1) |
| `rounds` | 32 个 `RoundState(index, direction, x, y, z)` 逐轮状态 |
| `sin_error`, `cos_error` | Decimal:`输出/2^28 - 高精度参考`,带符号 |

常量:`SCALE=2**28`、`ATAN_TABLE_Q`(32 项)、`GAIN_INV_Q`、`PI_Q`、`HALF_PI_Q`。

`sinh_cosh` 返回 `HyperbolicResult`(frozen dataclass):

| 字段 | 含义 |
|---|---|
| `cosh_q`, `sinh_q` | Q4.28 整数输出(cosh/sinh × 2^28) |
| `z_q` | 回显的输入角度 |
| `rounds` | 34 个 `HyperbolicRoundState` 逐轮状态(见下) |
| `cosh_error`, `sinh_error` | Decimal:`输出/2^28 - 高精度参考`,带符号 |

每个 `HyperbolicRoundState` 记录 `round_no`、`index`(移位索引 i)、`repeated`
(是否为重复轮)、`shift`(实际移位量)、`direction`、`x_shift`/`y_shift`
(本轮实际使用的移位后操作数,舍入记录)及更新后的 `x`/`y`/`z`。

双曲常量:`ATANH_TABLE_Q`(按下标 i=1..32 直接索引,0 号位为占位)、
`HYP_SCHEDULE`(34 项 `(i, repeated)` 调度)、`HYP_REPEAT_INDICES=(4, 13)`、
`HYP_GAIN_INV_Q`(含重复轮的增益逆)、`Z_MAX_Q=SCALE`(输入域 |z| ≤ 1)。

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
- 所有异常继承自 `CordicError`。`sin_cos`/`sinh_cosh` 是纯函数:调用失败不留任何部分状态。
- 核心与参考均**不调用** `math.sin/cos/sinh/cosh/exp`(有测试守护)。

## 双曲旋转契约

- **入口**:`sinh_cosh(z_q)`,只验收 |z| ≤ 1(|z_q| ≤ SCALE = 2^28);超出抛
  `InputRangeError`(可定位,拒绝而非钳位/饱和)。
- **调度**:双曲微旋转从 i=1 开始,共 32 个索引;按 i_next=3i+1 规则,
  i=4、13 两轮各重复一次(40 > 32 不在范围内),调度共 34 个微旋转。
- **增益补偿**:`HYP_GAIN_INV_Q` 是全部 34 个实际执行微旋转(含重复轮)的
  1/sqrt(1-2^-2i) 连乘逆,由 Decimal 高精度(80 位)生成并按 ties-to-even
  冻结到 Q4.28;重复轮计入增益,有测试独立复算守护。
- **常数表**:`ATANH_TABLE_Q` 由 atanh Taylor 级数(Decimal,80 位)生成并冻结,
  全部标准库离线计算,不取网络表。
- **移位/溢出**:与圆旋转同一约定——算术右移向负无穷取整,每次寄存器更新做
  signed 32-bit 检查,溢出抛 `FixedPointOverflowError`,绝不饱和后仍报正确。
- **参考**:`sinh_cosh_ref` 用 Decimal 指数定义 cosh=(e^z+e^-z)/2、
  sinh=(e^z-e^-z)/2(60 位精度)独立计算,不调用被测核心。

## 误差口径(浮点结果如何解读)

- `sin_error`/`cos_error` 是**带符号绝对误差**,单位是实数(弧度域的函数值),
  由 `输出/2^28` 减去 60 位精度 Decimal Taylor 参考得到;乘 `SCALE` 得 ulp 数。
- 参考本身经 Decimal 范围缩减 + Taylor 级数独立计算,**不调用被测核心**;
  参考对量化角(而非数学角)求值,故在 pi/2 等处残差 ~1 ulp 属预期。
- 解析误差上界取 **80 ulp**(增益量化 1 ulp + 32 轮移位截断经 ~1.647 增益
  放大 + 表量化),测试以此断言;全范围 2001 点扫描实测最大约 **13.7 ulp**
  (≈5.1e-8)。扫描是采样而非穷举,demo 中已注明。
- 双曲入口解析上界取 **120 ulp**(34 个微旋转、增益 ~1.2075);对 0、±1 边界及
  重复轮(i=4、13)±4 ulp 邻域加 400 点网格共 423 个配置实测最大约 **11.0 ulp**
  (≈4.1e-8),测试与 `demo_iteration.py` 均按配置报告实测最大值。
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

## 已知限制

- 仅实现本题契约的旋转模式 sin/cos 与双曲 cosh/sinh(|z| ≤ 1);不含向量模式、
  arctan、tanh、exp/ln 等完整超越函数库,不宣称任何行业标准兼容。
- 双曲入口只验收 |z| ≤ 1;更大角度需调用方先自行规约(本库拒绝而非回绕)。
- Q4.28 角度上限 ±8 rad;更大角度需调用方先自行规约(本库拒绝而非回绕)。
- demo 扫描为固定网格采样,非 2^32 穷举;穷举不在 8 秒演示预算内。

## 命令

```bash
python -m unittest discover -s tests -v   # 运行全部测试(55 项,含基线回归)
python demo.py                            # 圆旋转演示,约 1 秒内完成
python demo_iteration.py                  # 双曲演示,远小于 90 秒:逐轮轨迹 + 实测误差 + 拒绝
```

## 目录结构

```
cordic_q428/
  __init__.py      公共 API 导出
  errors.py        异常层级(CordicError 及子类)
  fixedpoint.py    Q4.28 原语:sar、量化、32-bit 检查、输入验证
  tables.py        Decimal 高精度生成 atan 表 / 增益逆 / pi 常量
  reference.py     独立 Decimal 参考(oracle):Taylor sin/cos、指数定义 sinh/cosh
  cordic.py        32 步旋转模式 CORDIC 核心(圆)
  hyperbolic.py    双曲旋转模式 CORDIC 核心(i=1 起,i=4/13 重复)
tests/             unittest:契约、核心、参考独立性、双曲调度与误差
demo.py            圆旋转固定输入演示
demo_iteration.py  双曲固定输入演示(逐轮轨迹、实测最大误差、拒绝演示)
```

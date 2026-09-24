# cordic_q428 — 定点 CORDIC:旋转模式 sin/cos 与向量模式 atan2/模长

离线嵌入式数值工程用可复用模块:signed Q4.28 定点、32 步 CORDIC,
覆盖两种模式——

* **旋转模式** `sin_cos`:给定角度输出 sin/cos 整数、逐轮状态与相对
  高精度参考的误差;
* **向量模式** `atan2_mag`:给定向量 (y, x) 输出 `atan2(y, x)` 角与模长
  `sqrt(x²+y²)`、象限偏移、饱和标志、逐轮状态与误差。

纯 Python 标准库实现,无任何第三方依赖,完全离线运行。

## 环境

- Python 3.14.7(标准库即可,`requirements.txt` 为空约定,无需安装任何依赖)
- Linux 原生运行;不需要网络、账号、数据库、Docker 或硬件

## 接口

```python
from cordic_q428 import (
    sin_cos, sin_cos_radians, sin_cos_ref,           # 旋转模式
    atan2_mag, atan2, magnitude, atan2_mag_radians,  # 向量模式
    atan2_mag_ref,
)

# 旋转模式
r = sin_cos(angle_q)          # angle_q: int,Q4.28 弧度角 [-2**31, 2**31-1]
r = sin_cos_radians(1.0)      # 便捷封装:有限 float 弧度 -> Q4.28
s, c = sin_cos_ref(angle_q)   # 独立 Decimal Taylor 参考(oracle),Decimal

# 向量模式(参数顺序遵循 atan2 惯例:纵向 y 在前)
v = atan2_mag(y_q, x_q)       # -> Atan2Result
a = atan2(y_q, x_q)           # 只要角度(Q4.28 int)
m = magnitude(y_q, x_q)       # 只要模长(Q4.28 int,超出格式则饱和)
v = atan2_mag_radians(-0.5, 1.25)          # 便捷封装:有限 float 坐标
a, m = atan2_mag_ref(y_q, x_q)             # 独立 Decimal 参考(oracle)
```

### `CordicResult`(旋转模式)

| 字段 | 含义 |
|---|---|
| `sin_q`, `cos_q` | Q4.28 整数输出(sin/cos × 2^28) |
| `reduced_q` | 象限规约到 [-pi/2, pi/2] 后的角度 |
| `sign_flip` | 规约提取的象限符号(+1/-1) |
| `rounds` | 32 个 `RoundState(index, direction, x, y, z)` 逐轮状态 |
| `sin_error`, `cos_error` | Decimal:`输出/2^28 - 高精度参考`,带符号 |

### `Atan2Result`(向量模式)

| 字段 | 含义 |
|---|---|
| `angle_q` | Q4.28 角度 `atan2(y, x)`,落在 `[-pi, pi]` |
| `mag_q` | Q4.28 模长 `sqrt(x²+y²)`;超格式上限时钳到 `INT32_MAX` |
| `saturated` | 模长是否发生饱和(bool) |
| `angle_offset_q` | 象限预处理施加的偏移:`0` / `+PI_Q` / `-PI_Q` |
| `rounds` | 32 个 `RoundState`;零向量时 32 个空转(`direction=0`) |
| `angle_error`, `mag_error` | Decimal,相对独立 Decimal oracle 的带符号误差;饱和时 `mag_error` 为钳制缺口(负值) |

`angle` / `mag` / `sin` / `cos` 是把整数结果除以 `SCALE` 的 float 视图。

常量:`SCALE=2**28`、`FRAC_BITS=28`、`INT32_MIN/MAX`、`ATAN_TABLE_Q`(32 项)、
`GAIN_INV_Q`、`PI_Q`、`HALF_PI_Q`、`TWO_PI_Q`、`ITERATIONS=32`。
新增定点原语:`mul_q`(Q4.28 乘,ties-to-even)、`rshift_even`(ties-to-even
右移)、`saturate_int32`(钳位并返回是否钳制)。

## 向量模式算法

1. **象限预处理**。`x < 0` 时把 (x, y) 同时取反(等价于旋转 pi),角度累加器
   初值取 `z0 = +pi`(y≥0)或 `-pi`(y<0),使迭代只在右半平面 `x≥0` 进行。
   取反用 Python 任意精度整数完成,因此坐标含 `INT32_MIN` 也不会溢出回绕。
2. **模长归一化(左移)**。atan2 对各向同性缩放不变:把折叠后两坐标同时左移
   k 位,使较大分量 ≥ 1.0,这样 1 ulp 的极小向量也能驱动全部 32 轮(否则
   算术右移的向负无穷取整会让 `-1 >> i == -1`,尾部轮次无法收敛)。模长在
   末尾按同一 2^k 除回。
3. **32 步整数移位向量迭代**,判决 `d = -sign(y)`(y<0 取 +1),更新公式与
   旋转模式完全相同;y 被驱向 0,z 从象限偏移收敛到 atan2 角。
4. **增益补偿**。旋转使横向寄存器放大 CORDIC 增益 Gₙ≈1.647;迭代结束后用
   一次 ties-to-even 的 Q4.28 乘 `GAIN_INV_Q` 去除,再 ties-to-even 右移 k 位
   还原归一化,最后饱和到 signed 32-bit。

### 特殊情形的明确定义

- **零向量 (0, 0)**:`angle=0`、`mag=0`、`saturated=False`,32 轮全部空转
  (`direction=0`,寄存器恒 0),误差为 0。整数输入没有 signed-zero,这是文档化
  约定(与 `math.atan2(+0.0, ...)` 同族,显式给出而非依赖浮点符号位)。
- **负 x 轴 (0, x<0)**:`angle = +pi`(同理取 +pi 分支)。
- **含 `INT32_MIN` 的坐标**:预处理宽整数取负,合法,绝不报错或回绕。
- **模长超界**:输入域是整个 signed 32-bit 方框,角点模长可达 √2·8 ≈ 11.31。
  当真实模长超过 Q4.28 上限时,角度照常精确计算,**模长饱和到 `INT32_MAX`**,
  `saturated=True`,`mag_error` 报告(负的)钳制缺口。为保证角度不被中途钳制
  破坏,迭代中的 x/y 寄存器用任意精度整数承载(故 `rounds` 里的原始 x/y 在
  饱和场景可能超出 32 位,而 z 始终在范围内);只有最终提交值做范围检查/饱和。
  在恰为 `INT32_MAX` 的轴向输入上,数据路径有约 1 ulp 的正偏差会触发钳制,
  但钳制值本身就是正确的可表示最大值(`mag_error≈0`)。

## 数值契约与错误语义

- **格式**:signed Q4.28,1 ulp = 2^-28 ≈ 3.73e-9,坐标/角度范围 [-8.0, 8.0)。
- **移位语义**:右移为算术移位、向负无穷取整(`sar`,即 Python `>>`)。
  归一化还原用独立的 ties-to-even 舍入移位 `rshift_even`。
- **量化**:atan 表、增益逆由 Decimal 高精度级数(80 位)生成,按
  **ties-to-even** 量化;Q4.28 乘法 `mul_q` 同为 ties-to-even。
- **溢出**:旋转模式任何 signed 32-bit 越界抛 `FixedPointOverflowError`
  (携带 `where`/`value`)。向量模式不抛模长溢出,而是饱和并置标志(见上)。
- **类型**:`bool` 不是合法定点整数,抛 `InvalidArgumentTypeError`;
  NaN/±Infinity(float 入口)抛 `NonFiniteValueError`。非法输入一律拒绝,
  不静默纠正;两个坐标先全部校验再计算,失败不留半更新状态(纯函数)。
- 所有异常继承自 `CordicError`。核心与参考均**不调用** `math.sin/cos/tan/atan2/hypot`
  (有测试守护;`math` 仅用于测试/demo 中独立核对)。

## 误差口径

- `*_error` 是**带符号绝对误差**,单位是实数;乘 `SCALE` 得 ulp 数。参考由
  Decimal 独立计算(旋转:范围缩减 + Taylor;向量:半角恒等式把 atan 参数拉到
  ≤0.5 后 Taylor,模长用 `Decimal.sqrt`),**不调用被测核心**,且对量化后的
  Q4.28 输入求值。
- **旋转模式**:解析安全上界 80 ulp;全范围 2001 点扫描实测最大约 **13.7 ulp**。
- **向量模式**(32 轮移位截断经 ~1.647 增益放大 + 增益表量化 1 ulp):解析安全
  上界取 **48 ulp**,测试据此断言;20 极环 × 401 角的确定性扫描实测角度最大约
  **4.7 ulp**、非饱和模长最大约 **14.4 ulp**。极小向量(分量 1 ulp 量级)经
  归一化后角度同样满足该界。扫描为采样而非穷举。
- 由于移位向负无穷取整,+q/-q 输出不保证精确奇对称,实测偏差为个位数 ulp。

## 输入上限

- 旋转 `angle_q` / 向量 `y_q`,`x_q`:signed 32-bit 整数,即
  [-2147483648, 2147483647](实域 [-8.0, 8.0))。
- `sin_cos_radians` / `atan2_mag_radians`:有限 float,每个量 |·| < 8.0;
  8.0 不可表示,触发拒绝。
- 迭代数固定 32;旋转模式规约循环预算 8 步,耗尽抛 `RangeReductionError`。
- 向量模长可超过 8.0(对角点约 11.31):角度始终有效,模长按上文饱和。

## 设计取舍

- 旋转规约、向量象限处理都用整数加减而非浮点取模:全程定点、可复现。
- 向量模式增益补偿放在迭代**之后**而非之前:预乘 1/Gₙ 会把 1 ulp 输入舍成 0。
- 用左移归一化换取全输入域(含极小向量)的角度精度,模长末尾除回 2^k。
- 迭代中 x/y 用宽整数、只钳最终模长:中途钳位会污染角度;饱和是明确的输出
  状态而非静默截断。
- 每次 z 更新仍做 32-bit 检查,精确定位真正的角度寄存器溢出。
- 误差以 Decimal 报告,避免在报告中再引入一次舍入。

## 已知限制

- 仅实现 sin/cos(旋转)与 atan2/模长(向量);不含双曲 CORDIC、sinh/cosh、
  反正弦/余弦,不宣称任何行业标准兼容。
- 字长/小数位/迭代数固定为 Q4.28 / 32 步;不支持任意字长,不提供逐步误差
  区间证书或提前停止认证(本轮迭代明确不做)。
- 模长超过 `INT32_MAX/SCALE ≈ 7.999999996` 时饱和,无法表示更大模长。
- 零向量角定义为 0、负 x 轴 (0, x<0) 角定义为 +pi,是无 signed-zero 整数
  输入下的文档化约定。
- demo 扫描为固定网格/极环采样,非穷举;穷举不在演示预算内。

## 命令

```bash
python -m unittest discover -s tests -v   # 运行全部测试(83 项)
python demo.py                            # 约 1 秒:两种模式正常结果 +
                                          # 扫描 + 边界 + 真实拒绝演示
```

## 目录结构

```
cordic_q428/
  __init__.py      公共 API 导出
  errors.py        异常层级(CordicError 及子类)
  fixedpoint.py    Q4.28 原语:sar、量化、32-bit 检查、mul_q、rshift_even、饱和
  tables.py        Decimal 高精度生成 atan 表 / 增益逆 / pi 常量
  reference.py     独立 Decimal oracle:sin/cos Taylor;atan2 半角+Taylor、Decimal.sqrt
  cordic.py        32 步旋转模式与向量模式 CORDIC 核心
tests/
  test_contract.py    契约:输入校验、移位/量化规则
  test_cordic.py      旋转模式核心
  test_reference.py   参考独立性
  test_vectoring.py   向量模式:解析值、四象限、零向量、INT32_MIN、饱和、失败用例
demo.py               固定输入演示(两种模式 + 边界 + 拒绝)
```

"""本题库统一的错误类型。

错误语义（公开契约）：
- InvalidInputError  : 输入类型非法（bool、float、NaN/Infinity、非 int 等），
                       属于调用方可定位的编程错误，同时是 TypeError。
- AngleRangeError    : 角度整数超出 signed32 Q4.28 可表示范围，同时是 ValueError。
- CordicOverflowError: 迭代中间量或最终结果超出 signed32，同时是 OverflowError。

所有错误消息均包含出错位置（参数名或迭代轮次），不静默纠正输入。
"""


class CordicError(Exception):
    """本题库所有错误的基类。"""


class InvalidInputError(CordicError, TypeError):
    """输入类型非法：bool、float（含 NaN/Infinity）、非整数类型等。"""


class AngleRangeError(CordicError, ValueError):
    """角度原始整数超出 signed32 范围 [-2**31, 2**31 - 1]。"""


class CordicOverflowError(CordicError, OverflowError):
    """CORDIC 迭代中间量或结果超出 signed32 范围。"""

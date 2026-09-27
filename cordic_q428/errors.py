"""Exception hierarchy for the fixed-point CORDIC library.

All exceptions raised by this package derive from :class:`CordicError`, so
callers may catch every library-defined error with a single base class.
"""

from __future__ import annotations


class CordicError(Exception):
    """Base class for every error raised by ``cordic_q428``."""


class FixedPointOverflowError(CordicError, OverflowError):
    """Raised when a signed 32-bit Q4.28 value leaves ``[-2**31, 2**31-1]``.

    The CORDIC datapath never wraps silently: every register add/subtract and
    every interface value carrying Q4.28 data is range-checked.  ``where``
    identifies the operation or parameter name so the failure is locatable.
    """

    def __init__(self, message: str, *, value: object | None = None, where: str | None = None):
        super().__init__(message)
        self.value = value
        self.where = where

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        msg = super().__str__()
        loc = []
        if self.where is not None:
            loc.append(f"where={self.where!r}")
        if self.value is not None:
            loc.append(f"value={self.value!r}")
        return f"{msg} ({', '.join(loc)})" if loc else msg


class InvalidArgumentTypeError(CordicError, TypeError):
    """Raised when an argument has the wrong Python type.

    ``bool`` is rejected everywhere an integer is required (``True``/``False``
    are silently ``int`` in Python but are not valid fixed-point samples).
    """


class NonFiniteValueError(CordicError, ValueError):
    """Raised when a real-valued numeric input is NaN or +/-infinity."""


class RangeReductionError(CordicError, ArithmeticError):
    """Raised if quadrant reduction cannot converge within its fixed budget."""


class ToleranceNotGuaranteedError(CordicError, ArithmeticError):
    """Raised when the requested absolute error tolerance cannot be certified.

    Early termination may only be claimed when the library's a-priori
    fixed-point error bound (which includes residual angle, shift
    truncation and quantization) is below the caller's tolerance.  If even
    the best attainable bound over the 32 rounds is larger, this exception
    is raised: the tolerance is never silently relaxed.

    Attributes:
        requested_tolerance: the tolerance the caller asked for (real units).
        best_bound: the smallest provable bound attainable at 32 rounds.
        steps: number of rounds executed while proving impossibility (32).
    """

    def __init__(
        self,
        message: str,
        *,
        requested_tolerance: object | None = None,
        best_bound: object | None = None,
        steps: object | None = None,
    ):
        super().__init__(message)
        self.requested_tolerance = requested_tolerance
        self.best_bound = best_bound
        self.steps = steps

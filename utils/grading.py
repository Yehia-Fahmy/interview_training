from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str = ""


def assert_true(name: str, cond: bool, message: str = "") -> CheckResult:
    return CheckResult(name=name, passed=bool(cond), message=message if not cond else "")


def assert_equal(name: str, a: Any, b: Any) -> CheckResult:
    ok = a == b
    msg = "" if ok else f"Expected {b!r}, got {a!r}"
    return CheckResult(name=name, passed=ok, message=msg)


def assert_less(name: str, value: float, threshold: float) -> CheckResult:
    ok = value < threshold
    msg = "" if ok else f"{value} !< {threshold}"
    return CheckResult(name=name, passed=ok, message=msg)


def run_checks(*checks: Callable[[], CheckResult]) -> None:
    results = [chk() for chk in checks]
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        line = f"[{status}] {r.name}"
        if r.message and not r.passed:
            line += f" — {r.message}"
        print(line)
    print(f"Summary: {passed}/{total} checks passed")
    if passed != total:
        raise SystemExit(1)



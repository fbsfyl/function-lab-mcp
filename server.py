"""Function Lab - a local MCP server for function analysis.

Tools:
  - analyze_function    : domain, parity, roots, extrema, monotonicity, concavity, asymptotes
  - plot_functions      : render one or more functions to PNG
  - calculus            : derivative / integral / limit / taylor / simplify
  - solve_equation      : symbolic + numeric equation (or system) solving
  - compare_functions   : side-by-side feature table + overlay plot

Transport: stdio. Requires: mcp, sympy, numpy, scipy, matplotlib.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from mcp.server.fastmcp import FastMCP
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

mcp = FastMCP("function-lab", instructions="本地函数分析：性质、作图、微积分、方程求解、函数族对比。")

OUTPUT_DIR = os.environ.get(
    "FUNCTION_LAB_OUTPUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------- parsing ----

_FUNCS: dict[str, Any] = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log, "ln": sp.log,
    "sqrt": sp.sqrt, "cbrt": sp.cbrt,
    "abs": sp.Abs, "Abs": sp.Abs,
    "sign": sp.sign, "floor": sp.floor, "ceil": sp.ceiling,
    "pi": sp.pi, "e": sp.E, "E": sp.E, "oo": sp.oo, "inf": sp.oo,
    "Gamma": sp.gamma, "erf": sp.erf,
}

_ALLOWED_CHARS = re.compile(r"^[0-9A-Za-z_+\-*/()., \t^]*$")
_TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)


def _sym(var: str) -> sp.Symbol:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", var):
        raise ValueError(f"非法变量名: {var!r}")
    if var in _FUNCS:
        raise ValueError(f"变量名与内置函数冲突: {var!r}")
    return sp.Symbol(var, real=True)


def _parse_multi(expression: str, variables: Sequence[str]) -> sp.Expr:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("表达式不能为空")
    if not _ALLOWED_CHARS.match(expression):
        raise ValueError("表达式含非法字符，仅允许数字、字母、+-*/().,^ 与空格")
    symbols = {name: _sym(name) for name in variables}
    try:
        expr = parse_expr(
            expression.strip(),
            local_dict={**_FUNCS, **symbols},
            transformations=_TRANSFORMS,
        )
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"无法解析表达式 {expression!r}: {exc}") from exc
    if not isinstance(expr, sp.Expr):
        raise ValueError(f"表达式不是有效的数学式: {expression!r}")
    extra = expr.free_symbols - set(symbols.values())
    if extra:
        raise ValueError(f"表达式含未知符号: {', '.join(sorted(map(str, extra)))}")
    return expr


def _parse(expression: str, var: str = "x") -> tuple[sp.Expr, sp.Symbol]:
    return _parse_multi(expression, [var]), _sym(var)


def _num(value: Any) -> float | None:
    try:
        approx = sp.N(value, 20)
        if approx in (sp.oo, -sp.oo, sp.zoo, sp.nan):
            return None
        comp = complex(approx)
    except Exception:  # noqa: BLE001
        return None
    if abs(comp.imag) > 1e-9 * max(1.0, abs(comp.real)):
        return None
    out = comp.real
    return out if math.isfinite(out) else None


def _is_infinite(value: sp.Expr) -> bool:
    if value in (sp.oo, -sp.oo, sp.zoo):
        return True
    try:
        return bool(value.is_infinite)
    except Exception:  # noqa: BLE001
        return False


def _domain_to_text(dom: Any) -> str:
    if dom == sp.S.Reals:
        return "(-∞, +∞)"
    if isinstance(dom, sp.Interval):
        left = "-∞" if dom.start == -sp.oo else str(dom.start)
        right = "+∞" if dom.end == sp.oo else str(dom.end)
        return f"{'(' if dom.left_open else '['}{left}, {right}{')' if dom.right_open else ']'}"
    if isinstance(dom, sp.Union):
        return " ∪ ".join(_domain_to_text(arg) for arg in dom.args)
    if isinstance(dom, sp.FiniteSet):
        return "{" + ", ".join(str(v) for v in dom.args) + "}"
    return str(dom)


def _fmt(value: float | None) -> str:
    if value is None:
        return "undefined"
    if abs(value) >= 1e5 or (value != 0 and abs(value) < 1e-4):
        return f"{value:.6g}"
    return f"{round(value, 6):g}"


def _real_roots(expr: sp.Expr, x: sp.Symbol, lo: float, hi: float) -> list[float]:
    roots: list[float] = []
    try:
        sols = sp.solve(sp.Eq(expr, 0), x)
    except Exception:  # noqa: BLE001
        sols = []
    for sol in sols:
        if isinstance(sol, sp.FiniteSet):
            candidates = list(sol)
        else:
            candidates = [sol]
        for cand in candidates:
            val = _num(cand)
            if val is not None and lo <= val <= hi and cand.is_real is not False:
                roots.append(val)
    return sorted({round(r, 9) for r in roots})


def _refine_root(expr: sp.Expr, x: sp.Symbol, left: float, right: float) -> float | None:
    try:
        from scipy.optimize import brentq

        return float(brentq(lambda t: float(sp.N(expr.subs(x, t))), left, right, maxiter=200))
    except Exception:  # noqa: BLE001
        return None


def _numeric_roots(expr: sp.Expr, x: sp.Symbol, lo: float, hi: float, limit: int = 40) -> list[float]:
    xs = np.linspace(lo, hi, 20001)
    ys = _sample(expr, x, xs)
    found: list[float] = []
    for i in range(len(xs) - 1):
        a, b = ys[i], ys[i + 1]
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        if a == 0:
            found.append(float(xs[i]))
        elif a * b < 0:
            root = _refine_root(expr, x, float(xs[i]), float(xs[i + 1]))
            if root is not None:
                value = _num(expr.subs(x, root))
                if value is not None and abs(value) < 1e-6:
                    found.append(root)
        if len(found) >= limit:
            break
    return sorted({round(v, 6) for v in found})


def _all_roots(expr: sp.Expr, x: sp.Symbol, lo: float, hi: float) -> list[float]:
    merged = sorted(_real_roots(expr, x, lo, hi) + _numeric_roots(expr, x, lo, hi))
    unique: list[float] = []
    for value in merged:
        if not unique or abs(value - unique[-1]) > 1e-6:
            unique.append(round(value, 6))
    return unique


def _sample(expr: sp.Expr, x: sp.Symbol, xs: np.ndarray) -> np.ndarray:
    try:
        func = sp.lambdify(x, expr, modules=["numpy"])
    except Exception:  # noqa: BLE001
        func = None
    if func is not None:
        with np.errstate(all="ignore"):
            try:
                ys = np.asarray(func(xs), dtype=complex)
                if ys.shape != xs.shape:
                    ys = np.broadcast_to(ys, xs.shape)
                ys = np.where(np.abs(ys.imag) < 1e-9, ys.real, np.nan)
                return ys.astype(float)
            except Exception:  # noqa: BLE001
                pass
    out = []
    for val in xs:
        try:
            res = complex(sp.N(expr.subs(x, val)))
            out.append(res.real if abs(res.imag) < 1e-9 else np.nan)
        except Exception:  # noqa: BLE001
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def _sign_at(expr: sp.Expr, x: sp.Symbol, at: float) -> int:
    try:
        val = complex(sp.N(expr.subs(x, at)))
    except Exception:  # noqa: BLE001
        return 0
    if abs(val.imag) > 1e-9:
        return 0
    return 1 if val.real > 1e-9 else (-1 if val.real < -1e-9 else 0)


def _midpoint(left: float, right: float) -> float:
    if math.isinf(left):
        return right - 1.0
    if math.isinf(right):
        return left + 1.0
    return (left + right) / 2


def _interval_label(left: float, right: float) -> str:
    def edge(value: float) -> str:
        if math.isinf(value):
            return "-∞" if value < 0 else "+∞"
        return _fmt(value)

    return f"({edge(left)}, {edge(right)})"


def _intervals(points: Iterable[float], lo: float, hi: float) -> list[tuple[float, float]]:
    edges = [lo] + sorted({p for p in points if lo < p < hi}) + [hi]
    return list(zip(edges[:-1], edges[1:]))


def _singularities(expr: sp.Expr, x: sp.Symbol) -> list[float]:
    out: list[float] = []
    try:
        from sympy.calculus.util import singularities

        found = singularities(expr, x, sp.S.Reals)
        items = list(found) if found is not None and not isinstance(found, sp.ConditionSet) else []
    except Exception:  # noqa: BLE001
        items = []
    # rational-function fallback: roots of the denominator
    try:
        num, den = sp.fraction(sp.together(expr))
        if den != 1:
            items += list(sp.solve(sp.Eq(den, 0), x))
    except Exception:  # noqa: BLE001
        pass
    for item in items:
        val = _num(item)
        if val is not None:
            out.append(val)
    return sorted({round(v, 9) for v in out})


def _domain_label(expr: sp.Expr, x: sp.Symbol) -> tuple[str, list[float]]:
    singular = _singularities(expr, x)
    try:
        from sympy.calculus.util import continuous_domain

        dom = continuous_domain(expr, x, sp.S.Reals)
        text = _domain_to_text(dom)
    except Exception:  # noqa: BLE001
        text = "无法解析定义域，请结合图像判断"
    return text, singular


def _asymptotes(expr: sp.Expr, x: sp.Symbol, singular: Sequence[float]) -> dict[str, list[Any]]:
    vertical: list[str] = []
    for point in singular:
        left_lim = sp.limit(expr, x, point, "-")
        right_lim = sp.limit(expr, x, point, "+")
        if _is_infinite(left_lim) or _is_infinite(right_lim):
            vertical.append(f"x = {_fmt(point)}")

    horizontal: list[str] = []
    for direction, sign_text in ((sp.oo, "+∞"), (-sp.oo, "-∞")):
        try:
            lim = sp.limit(expr, x, direction)
        except Exception:  # noqa: BLE001
            continue
        val = _num(lim)
        if val is not None and abs(val) < 1e6:
            horizontal.append(f"y = {_fmt(val)} (x → {sign_text})")

    oblique: list[str] = []
    if not horizontal:
        for direction, sign_text in ((sp.oo, "+∞"), (-sp.oo, "-∞")):
            try:
                m_lim = sp.limit(expr / x, x, direction)
                m = _num(m_lim)
                if m is None or abs(m) < 1e-9 or abs(m) > 1e6:
                    continue
                b = _num(sp.limit(expr - m * x, x, direction))
                if b is None:
                    continue
                oblique.append(f"y = {_fmt(m)}x {'+' if b >= 0 else '-'} {_fmt(abs(b))} (x → {sign_text})")
            except Exception:  # noqa: BLE001
                continue
    return {"vertical": vertical, "horizontal": horizontal, "oblique": oblique}


def _parity(expr: sp.Expr, x: sp.Symbol) -> str:
    try:
        simplified = sp.simplify(expr.subs(x, -x))
        if sp.simplify(simplified - expr) == 0:
            return "even"
        if sp.simplify(simplified + expr) == 0:
            return "odd"
    except Exception:  # noqa: BLE001
        pass
    return "neither"


def _period(expr: sp.Expr, x: sp.Symbol) -> str | None:
    try:
        from sympy.calculus.util import periodicity

        per = periodicity(expr, x)
        if per is not None and per.is_finite:
            return str(per)
    except Exception:  # noqa: BLE001
        pass
    return None


def _critical_points(expr: sp.Expr, x: sp.Symbol, lo: float, hi: float) -> list[float]:
    deriv = sp.diff(expr, x)
    points = _real_roots(deriv, x, lo, hi)
    if points:
        return points
    # numeric fallback: sign changes of f' refined by brentq
    xs = np.linspace(lo, hi, 4001)
    vals = _sample(deriv, x, xs)
    found: list[float] = []
    for i in range(len(xs) - 1):
        a, b = vals[i], vals[i + 1]
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        if a == 0:
            found.append(float(xs[i]))
        elif a * b < 0:
            root = _refine_root(deriv, x, float(xs[i]), float(xs[i + 1]))
            found.append(root if root is not None else float(xs[i]))
    return sorted({round(v, 6) for v in found})


def _extrema(expr: sp.Expr, x: sp.Symbol, lo: float, hi: float) -> list[dict[str, Any]]:
    deriv2 = sp.diff(expr, x, 2)
    result: list[dict[str, Any]] = []
    for point in _critical_points(expr, x, lo, hi):
        y = _num(expr.subs(x, point))
        if y is None:
            continue
        second = _num(deriv2.subs(x, point))
        if second is None:
            kind = "critical"
        elif second > 1e-9:
            kind = "local_min"
        elif second < -1e-9:
            kind = "local_max"
        else:
            kind = "saddle_or_inflection"
        result.append({"x": round(point, 9), "y": round(y, 9), "type": kind})
    return result


def _monotonic(expr: sp.Expr, x: sp.Symbol, singular: Sequence[float], lo: float, hi: float) -> list[dict[str, Any]]:
    deriv = sp.diff(expr, x)
    breaks = list(singular) + _critical_points(expr, x, lo, hi)
    out: list[dict[str, Any]] = []
    for left, right in _intervals(breaks, lo, hi):
        mid = _midpoint(left, right)
        sign = _sign_at(deriv, x, mid)
        behavior = {1: "increasing", -1: "decreasing", 0: "flat/undefined"}[sign]
        out.append({"interval": _interval_label(left, right), "behavior": behavior})
    return out


def _concavity(expr: sp.Expr, x: sp.Symbol, singular: Sequence[float], lo: float, hi: float) -> list[dict[str, Any]]:
    deriv2 = sp.diff(expr, x, 2)
    breaks = list(singular) + _real_roots(deriv2, x, lo, hi)
    out: list[dict[str, Any]] = []
    for left, right in _intervals(breaks, lo, hi):
        mid = _midpoint(left, right)
        sign = _sign_at(deriv2, x, mid)
        shape = {1: "concave_up", -1: "concave_down", 0: "flat/undefined"}[sign]
        out.append({"interval": _interval_label(left, right), "shape": shape})
    return out


# ------------------------------------------------------------ output utils ---

def _figure_path(name_hint: str, output_dir: str | None = None) -> str:
    folder = output_dir or OUTPUT_DIR
    os.makedirs(folder, exist_ok=True)
    safe = re.sub(r"[^0-9A-Za-z_-]+", "_", name_hint).strip("_") or "plot"
    index = 1
    while True:
        path = os.path.join(folder, f"{safe}_{index}.png")
        if not os.path.exists(path):
            return path
        index += 1


def _save(fig, name_hint: str, output_dir: str | None = None) -> str:
    path = _figure_path(name_hint, output_dir)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    return path


def _empty_axes(title: str):
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.set_title(title)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


# ------------------------------------------------------------------ tools ----

@mcp.tool()
def analyze_function(expression: str, variable: str = "x", scope: float = 20.0) -> dict:
    """全面分析一个一元函数。

    返回定义域、化简式、一阶/二阶导数、奇偶性、周期、实根、极值点、
    单调区间、凹凸区间、渐近线，以及在 ±scope 范围内的数值值域。

    Args:
        expression: 函数表达式，例如 "x^3 - 3x + 1"、"sin(x)/x"、"exp(-x^2)"。
        variable: 自变量名，默认 "x"。
        scope: 数值扫描半径（同时用于渐近线/值域估算），默认 20。
    """
    expr, x = _parse(expression, variable)
    lo, hi = -abs(scope), abs(scope)
    domain_text, singular = _domain_label(expr, x)
    deriv = sp.diff(expr, x)
    deriv2 = sp.diff(expr, x, 2)
    xs = np.linspace(lo, hi, 6001)
    ys = _sample(expr, x, xs)
    finite = ys[np.isfinite(ys)]

    return {
        "input": expression,
        "simplified": str(sp.simplify(expr)),
        "latex": sp.latex(expr),
        "domain": domain_text,
        "parity": _parity(expr, x),
        "period": _period(expr, x),
        "derivative": str(sp.simplify(deriv)),
        "second_derivative": str(sp.simplify(deriv2)),
        "roots": _all_roots(expr, x, lo, hi),
        "extrema": _extrema(expr, x, lo, hi),
        "monotonic": _monotonic(expr, x, singular, lo, hi),
        "concavity": _concavity(expr, x, singular, lo, hi),
        "singularities": singular,
        "asymptotes": _asymptotes(expr, x, singular),
        "range_numeric": {
            "min": round(float(np.min(finite)), 9) if finite.size else None,
            "max": round(float(np.max(finite)), 9) if finite.size else None,
            "scope": [lo, hi],
        },
        "note": "区间与值域基于符号推导 + 数值采样，含振荡/间断的函数请以图像为准。",
    }


@mcp.tool()
def plot_functions(
    expressions: Sequence[str],
    x_min: float = -10.0,
    x_max: float = 10.0,
    y_min: float | None = None,
    y_max: float | None = None,
    title: str | None = None,
    variable: str = "x",
    samples: int = 3000,
    mark_points: Sequence[Sequence[float | str]] | None = None,
    output_dir: str | None = None,
) -> dict:
    """把一到多个函数画在同一张图上并保存为 PNG。

    Args:
        expressions: 表达式列表，例如 ["x^2", "2x", "sin(x)"]。
        x_min/x_max: 横轴范围。
        y_min/y_max: 纵轴范围，留空自动。
        title: 图标题。
        variable: 自变量名。
        samples: 采样点数，越大越平滑。
        mark_points: 需要标注的点，格式 [[x, y], ...] 或 [[x, y, "标签"], ...]。
        output_dir: 图片保存目录，默认服务器 output 目录。
    """
    if not isinstance(expressions, (list, tuple)):
        expressions = [expressions]
    if not expressions:
        raise ValueError("至少提供一个表达式")
    if x_max <= x_min:
        raise ValueError("x_max 必须大于 x_min")

    xs = np.linspace(x_min, x_max, max(int(samples), 200))
    fig, ax = _empty_axes(title or "函数图像")
    colors = ["#2f6fed", "#e05252", "#2fa36b", "#b45fd8", "#e08a2f", "#0fa3b1"]

    for idx, text in enumerate(expressions):
        expr, x = _parse(text, variable)
        ys = _sample(expr, x, xs)
        if not np.isfinite(ys).any():
            continue
        ys = np.where(np.abs(ys) > 1e9, np.nan, ys)
        ax.plot(xs, ys, label=f"$y={sp.latex(expr)}$", color=colors[idx % len(colors)], linewidth=1.8)

    if mark_points:
        for item in mark_points:
            if len(item) < 2:
                continue
            px, py = float(item[0]), float(item[1])
            label = str(item[2]) if len(item) > 2 else f"({_fmt(px)}, {_fmt(py)})"
            ax.plot([px], [py], "o", color="#111", markersize=5)
            ax.annotate(label, (px, py), textcoords="offset points", xytext=(6, 8), fontsize=9)

    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(variable)
    ax.set_ylabel(f"f({variable})")
    if len(expressions) > 1 or mark_points:
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
    path = _save(fig, "plot", output_dir)
    return {"image_path": path, "x_range": [x_min, x_max], "expressions": list(expressions)}


@mcp.tool()
def calculus(
    expression: str,
    operation: str,
    variable: str = "x",
    lower: str | None = None,
    upper: str | None = None,
    point: str | None = None,
    order: int = 5,
) -> dict:
    """符号微积分运算。

    Args:
        expression: 函数表达式。
        operation: 取值 derivative | second_derivative | integrate | definite_integral
                   | limit | taylor | simplify | expand | factor。
        variable: 自变量名。
        lower/upper: 定积分上下限（可用 "0"、"-oo"、"oo"、"pi"）。
        point: limit 的趋近点，taylor 的展开点（默认 0）。
        order: taylor 展开阶数，默认 5。
    """
    expr, x = _parse(expression, variable)
    op = operation.strip().lower()

    def parse_value(text: str) -> sp.Expr:
        return parse_expr(
            str(text).strip(),
            local_dict={**_FUNCS, variable: x},
            transformations=_TRANSFORMS,
        )

    if op == "derivative":
        result = sp.simplify(sp.diff(expr, x))
        return {"operation": op, "result": str(result), "latex": sp.latex(result)}
    if op == "second_derivative":
        result = sp.simplify(sp.diff(expr, x, 2))
        return {"operation": op, "result": str(result), "latex": sp.latex(result)}
    if op == "integrate":
        result = sp.integrate(expr, x)
        return {
            "operation": op,
            "result": str(result),
            "latex": sp.latex(result),
            "check": str(sp.simplify(sp.diff(result, x) - expr)),
        }
    if op == "definite_integral":
        if lower is None or upper is None:
            raise ValueError("定积分需要 lower 与 upper")
        a, b = parse_value(lower), parse_value(upper)
        result = sp.integrate(expr, (x, a, b))
        return {
            "operation": op,
            "interval": [str(a), str(b)],
            "result": str(result),
            "numeric": _num(result),
            "latex": sp.latex(result),
        }
    if op == "limit":
        if point is None:
            raise ValueError("limit 需要 point，例如 point='0' 或 'oo'")
        try:
            result = sp.limit(expr, x, parse_value(point))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"极限计算失败: {exc}") from exc
        return {"operation": op, "point": point, "result": str(result), "numeric": _num(result)}
    if op == "taylor":
        at = parse_value(point) if point else sp.Integer(0)
        result = sp.series(expr, x, at, int(order) + 1).removeO()
        return {
            "operation": op,
            "point": str(at),
            "order": int(order),
            "result": str(sp.expand(result)),
            "latex": sp.latex(result),
        }
    if op in {"simplify", "expand", "factor", "cancel", "apart"}:
        result = getattr(sp, op)(expr)
        return {"operation": op, "result": str(result), "latex": sp.latex(result)}
    raise ValueError(
        "operation 需为 derivative / second_derivative / integrate / definite_integral / "
        "limit / taylor / simplify / expand / factor / cancel / apart"
    )


@mcp.tool()
def solve_equation(equations: Sequence[str] | str, variables: Sequence[str] | str = "x", numeric_scope: float = 50.0) -> dict:
    """求解方程或方程组，符号解优先，无符号解时给出数值根。

    Args:
        equations: 单个方程字符串或列表。支持 "x^2 - 4 = 0"、"x^2 - 4"、"x + y = 3"。
        variables: 变量名或变量名列表，默认 "x"。方程组需按顺序给出。
        numeric_scope: 数值扫描半径，默认 50。
    """
    if isinstance(equations, str):
        equations = [equations]
    if isinstance(variables, str):
        variables = [variables]

    parsed: list[sp.Expr] = []
    for text in equations:
        if "=" in text:
            left, right = text.split("=", 1)
            expr = _parse_multi(left, variables) - _parse_multi(right, variables)
        else:
            expr = _parse_multi(text, variables)
        parsed.append(expr)

    syms = [_sym(v) for v in variables]
    if len(parsed) > 1:
        try:
            sols = sp.solve(parsed, syms, dict=True)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"方程组求解失败: {exc}") from exc
        return {
            "system": list(equations),
            "variables": list(variables),
            "solutions": [{str(k): str(sp.nsimplify(v)) for k, v in sol.items()} for sol in sols],
            "numeric": [{str(k): _num(v) for k, v in sol.items()} for sol in sols],
        }

    expr, x = parsed[0], syms[0]
    symbolic: list[str] = []
    try:
        for sol in sp.solve(sp.Eq(expr, 0), x):
            symbolic.append(str(sol))
    except Exception:  # noqa: BLE001
        symbolic = []

    roots = _all_roots(expr, x, -abs(numeric_scope), abs(numeric_scope))

    return {
        "equation": equations[0],
        "variable": str(x),
        "symbolic_solutions": symbolic,
        "real_roots": roots,
        "numeric_scope": [-abs(numeric_scope), abs(numeric_scope)],
        "note": "real_roots 由符号解与数值扫描合并；符号解在 symbolic_solutions。",
    }


@mcp.tool()
def compare_functions(
    expressions: Sequence[str],
    x_min: float = -10.0,
    x_max: float = 10.0,
    variable: str = "x",
    title: str | None = None,
    output_dir: str | None = None,
) -> dict:
    """并排对比多个函数（函数族/参数族）：特征表 + 叠加图像。

    Args:
        expressions: 表达式列表，例如 ["x^2", "x^3", "2^x"]。
        x_min/x_max: 对比区间。
        variable: 自变量名。
        title: 图标题。
        output_dir: 图片保存目录，默认服务器 output 目录。
    """
    if isinstance(expressions, str):
        expressions = [expressions]
    if len(expressions) < 1:
        raise ValueError("至少提供一个表达式")

    xs = np.linspace(x_min, x_max, 3000)
    fig, ax = _empty_axes(title or "函数族对比")
    colors = ["#2f6fed", "#e05252", "#2fa36b", "#b45fd8", "#e08a2f", "#0fa3b1"]
    table: list[dict[str, Any]] = []

    for idx, text in enumerate(expressions):
        expr, x = _parse(text, variable)
        ys = _sample(expr, x, xs)
        ys = np.where(np.abs(ys) > 1e9, np.nan, ys)
        finite = ys[np.isfinite(ys)]
        ax.plot(xs, ys, label=f"$y={sp.latex(expr)}$", color=colors[idx % len(colors)], linewidth=1.8)
        singular = _singularities(expr, x)
        table.append(
            {
                "expression": text,
                "parity": _parity(expr, x),
                "period": _period(expr, x),
                "roots_in_range": _real_roots(expr, x, x_min, x_max),
                "extrema_in_range": _extrema(expr, x, x_min, x_max),
                "y_min_in_range": round(float(np.min(finite)), 6) if finite.size else None,
                "y_max_in_range": round(float(np.max(finite)), 6) if finite.size else None,
                "monotonic": _monotonic(expr, x, singular, x_min, x_max),
            }
        )

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(variable)
    ax.set_ylabel(f"y")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    path = _save(fig, "compare", output_dir)
    return {
        "image_path": path,
        "range": [x_min, x_max],
        "features": table,
        "note": "单调区间在给定区间内按数值采样判定。",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

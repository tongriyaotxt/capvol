"""Turn results into something a reviewer can read.

Everything is plain ``json`` plus a markdown table — no notebook required,
no hidden state.  Numbers carry their intervals; a volume without an
interval is a rumour.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .compare import MethodResult

__all__ = ["to_json", "to_markdown", "results_table"]


def _plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _plain(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def to_json(obj: Any, path: str | Path | None = None) -> str:
    s = json.dumps(_plain(obj), indent=2, ensure_ascii=False)
    if path is not None:
        Path(path).write_text(s, encoding="utf-8")
    return s


def _fmt(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    if x == 0:
        return "0"
    if abs(x) >= 1e-2 and abs(x) < 1e4:
        return f"{x:.4g}"
    return f"{x:.3e}"


def results_table(results: Sequence[MethodResult]) -> list[str]:
    has_conn = any(r.connectivity is not None for r in results)
    head = ["method", "vol(F)", "95% CI", "rho", "estimator"]
    if has_conn:
        head += ["components", "largest ratio"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for r in results:
        e = r.estimate
        cells = [
            r.name,
            _fmt(e.vol),
            f"[{_fmt(e.vol_ci[0])}, {_fmt(e.vol_ci[1])}]",
            _fmt(e.rho),
            e.method,
        ]
        if has_conn:
            c = r.connectivity
            cells += [str(c.n_components) if c else "-",
                      f"{c.largest_ratio:.3f}" if c else "-"]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def to_markdown(
    results: Sequence[MethodResult],
    path: str | Path | None = None,
    title: str = "Capability report",
    v_max: float | None = None,
    space_volume: float | None = None,
    notes: Iterable[str] = (),
) -> str:
    """Render a comparison as a self-contained markdown block."""
    est = results[0].estimate if results else None
    vm = v_max if v_max is not None else (est.v_max if est else float("nan"))
    vg = space_volume if space_volume is not None else (est.space_volume if est else float("nan"))

    out: list[str] = [f"# {title}", ""]
    out.append("## Setup")
    out.append("")
    out.append(f"- goal-space volume `vol(G)` = {_fmt(vg)}")
    out.append(f"- cost budget `V_max` = {_fmt(vm)}")
    if est is not None:
        out.append(f"- samples per estimate: {est.n}")
    out.append("")
    out.append("## Results")
    out.append("")
    out += results_table(results)
    out.append("")

    if len(results) >= 2:
        base = results[0]
        out.append("## Relative to `" + base.name + "`")
        out.append("")
        out.append("| method | vol ratio | delta |")
        out.append("|---|---|---|")
        bv = base.estimate.vol
        for r in results[1:]:
            ratio = r.estimate.vol / bv if bv > 0 else float("nan")
            out.append(f"| {r.name} | {_fmt(ratio)}x | {_fmt(r.estimate.vol - bv)} |")
        out.append("")

    if notes:
        out.append("## Notes")
        out.append("")
        for n in notes:
            out.append(f"- {n}")
        out.append("")

    out.append("---")
    out.append("")
    out.append(
        "`vol(F)` is measured, not claimed: `F = {g : V(g) < V_max}` and the "
        "estimate comes with a 95% interval. Compare rows only inside this "
        "table — a volume computed in another goal space at another budget is "
        "a different quantity."
    )
    text = "\n".join(out)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text

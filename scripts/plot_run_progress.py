#!/usr/bin/env python3
"""Plot live progress of an engine-native small-molecule run from events.jsonl.

Reads candidate_evaluated events directly from the engine-native events.jsonl
(no need to wait for legacy history.json / rounds.jsonl which are materialized
only at campaign end) and writes:

  plots/progress_<run>.png  -- 2x2 panel: vina trajectory, activity trajectory,
                              Pareto front (best so far), and a score scatter.
  plots/progress_<run>.csv  -- per-evaluation (iteration, vina, activity) rows.

Usage:
  python scripts/plot_run_progress.py <run_dir> [--out <plots_dir>]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, help="Engine run directory (contains events.jsonl).")
    p.add_argument("--out", type=Path, default=None, help="Output directory (default: <run_dir>/plots).")
    return p.parse_args(argv)


def load_evaluated_scores(run_dir: Path) -> list[dict]:
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        raise SystemExit(f"events.jsonl not found: {events_path}")
    rows: list[dict] = []
    with events_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event_type") != "candidate_evaluated":
                continue
            payload = ev.get("payload") or {}
            evaluation = payload.get("evaluation") or {}
            if evaluation.get("status") != "succeeded":
                continue
            metrics = evaluation.get("metrics") or {}
            vina = metrics.get("vina")
            activity = metrics.get("activity")
            if vina is None or activity is None:
                continue
            try:
                vina_f = float(vina)
                act_f = float(activity)
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(vina_f) and np.isfinite(act_f)):
                continue
            rows.append(
                {
                    "iteration": ev.get("iteration"),
                    "candidate_id": evaluation.get("candidate_id", payload.get("candidate_id", "")),
                    "smiles": (payload.get("candidate") or {}).get("canonical_key", ""),
                    "vina": vina_f,
                    "activity": act_f,
                }
            )
    return rows


def pareto_front(rows: list[dict]) -> list[dict]:
    """Non-dominated points for minimize vina, maximize activity."""
    kept: list[dict] = []
    for r in rows:
        dominated = False
        for other in rows:
            if (
                other["vina"] <= r["vina"] + 1e-12
                and other["activity"] >= r["activity"] - 1e-12
                and (
                    other["vina"] < r["vina"] - 1e-12
                    or other["activity"] > r["activity"] + 1e-12
                )
            ):
                dominated = True
                break
        if not dominated:
            kept.append(r)
    return kept


def dominates_2d(left, right, minimize=(True, False)) -> bool:
    better_or_equal = []
    strictly_better = []
    for lv, rv, is_min in zip(left, right, minimize):
        if is_min:
            better_or_equal.append(lv <= rv)
            strictly_better.append(lv < rv)
        else:
            better_or_equal.append(lv >= rv)
            strictly_better.append(lv > rv)
    return all(better_or_equal) and any(strictly_better)


def pareto_score_indices(points, minimize=(True, False)) -> list[int]:
    indices = []
    for idx, point in enumerate(points):
        if not any(
            other_idx != idx and dominates_2d(other, point, minimize)
            for other_idx, other in enumerate(points)
        ):
            indices.append(idx)
    return indices


def compute_hypervolume_2d(
    points, ref_point=(0.0, 5.0), minimize=(True, False)
) -> float:
    """Pareto hypervolume for (vina minimize, activity maximize)."""
    converted = []
    ref = (
        float(ref_point[0]) if minimize[0] else -float(ref_point[0]),
        float(ref_point[1]) if minimize[1] else -float(ref_point[1]),
    )
    for point in points:
        cp = (
            float(point[0]) if minimize[0] else -float(point[0]),
            float(point[1]) if minimize[1] else -float(point[1]),
        )
        if cp[0] < ref[0] and cp[1] < ref[1]:
            converted.append(cp)
    if not converted:
        return 0.0
    front = [converted[idx] for idx in pareto_score_indices(converted, (True, True))]
    front.sort(key=lambda item: item[0])
    volume = 0.0
    for idx, point in enumerate(front):
        next_x = front[idx + 1][0] if idx + 1 < len(front) else ref[0]
        volume += max(0.0, next_x - point[0]) * max(0.0, ref[1] - point[1])
    return float(volume)


def hypervolume_by_iteration(rows: list[dict]) -> list[tuple]:
    """Cumulative Pareto hypervolume after each evaluated iteration."""
    out: list[tuple] = []
    seen: list[tuple[float, float]] = []
    for r in rows:
        seen.append((r["vina"], r["activity"]))
        hv = compute_hypervolume_2d(seen)
        out.append((r["iteration"] if r["iteration"] is not None else len(out), hv))
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_evaluated_scores(args.run_dir)
    if not rows:
        raise SystemExit("No finite succeeded evaluations found in events.jsonl.")
    rows.sort(key=lambda r: (r["iteration"] is None, r["iteration"] or 0))
    out_dir = args.out or (args.run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_dir.name

    iters = [r["iteration"] if r["iteration"] is not None else i + 1 for i, r in enumerate(rows)]
    vinas = [r["vina"] for r in rows]
    acts = [r["activity"] for r in rows]
    front = pareto_front(rows)
    hv_curve = hypervolume_by_iteration(rows)

    # CSV
    csv_path = out_dir / f"progress_{run_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["iteration", "candidate_id", "smiles", "vina", "activity", "hypervolume"])
        writer.writeheader()
        for r, (_it, hv) in zip(rows, hv_curve):
            writer.writerow({**r, "hypervolume": hv})

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Small-molecule real run: {run_name}  (successful evals: {len(rows)})", fontsize=14)

    ax = axes[0][0]
    ax.plot(iters, vinas, "o-", color="#1f77b4", alpha=0.8)
    ax.set_xlabel("iteration"); ax.set_ylabel("vina (lower better)"); ax.set_title("Vina trajectory")
    ax.grid(True, alpha=0.3)

    ax = axes[0][1]
    ax.plot(iters, acts, "s-", color="#2ca02c", alpha=0.8)
    ax.set_xlabel("iteration"); ax.set_ylabel("activity (higher better)"); ax.set_title("Activity trajectory")
    ax.grid(True, alpha=0.3)

    ax = axes[0][2]
    hv_iters = [it for it, _hv in hv_curve]
    hv_vals = [_hv for _it, _hv in hv_curve]
    ax.plot(hv_iters, hv_vals, "^-", color="#9467bd", linewidth=2)
    ax.set_xlabel("iteration"); ax.set_ylabel("Pareto hypervolume")
    ax.set_title(f"Pareto hypervolume vs iteration (final={hv_vals[-1]:.3f})")
    ax.grid(True, alpha=0.3)

    ax = axes[1][0]
    fv = [r["vina"] for r in front]
    fa = [r["activity"] for r in front]
    ax.plot(fv, fa, "D-", color="#d62728", markersize=6)
    ax.scatter(vinas, acts, s=30, alpha=0.5, color="#7f7f7f")
    for r in front:
        ax.annotate(str(r["iteration"]), (r["vina"], r["activity"]), fontsize=7, alpha=0.8)
    ax.set_xlabel("vina"); ax.set_ylabel("activity"); ax.set_title(f"Pareto front (n={len(front)})")
    ax.grid(True, alpha=0.3)

    ax = axes[1][1]
    ax.scatter(vinas, acts, c=iters, cmap="viridis", s=50)
    ax.set_xlabel("vina"); ax.set_ylabel("activity"); ax.set_title("Score scatter (colored by iteration)")
    cb = plt.colorbar(ax.collections[0], ax=ax)
    cb.set_label("iteration")
    ax.grid(True, alpha=0.3)

    ax = axes[1][2]
    ax.plot(hv_iters, hv_vals, "o-", color="#9467bd")
    ax.set_xlabel("iteration"); ax.set_ylabel("hypervolume")
    ax.set_title("Hypervolume (marker per evaluation)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = out_dir / f"progress_{run_name}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    best_vina = min(vinas)
    best_act = max(acts)
    print(f"rows={len(rows)} best_vina={best_vina:.3f} best_activity={best_act:.3f} front={len(front)} final_hypervolume={hv_vals[-1]:.3f}")
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

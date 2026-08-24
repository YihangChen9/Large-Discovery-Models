#!/usr/bin/env python3
"""Plot the nanoGPT 80-iteration real-run trajectory in README campaign style.

Reads the run's model_based_buffer.jsonl (observed val_bpb per evaluated
state) and draws the same scatter + best-so-far step plot used in the
repository README ("NanoGPT LDM: LCB" figure).

Usage:
  python scripts/plot_nanogpt_campaign.py <run_dir> [--out <png path>]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]

COLOR = "#C2413B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="nanoGPT run directory (contains model_based_buffer.jsonl).")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path (default: <run_dir>/plots/nanogpt_lcb_trajectory.png).")
    parser.add_argument("--total-iterations", type=int, default=None, help="Expected total iterations (for interim labeling).")
    parser.add_argument("--run-name", type=str, default=None, help="Only plot rows with this run_name (filter out inherited buffer history).")
    return parser.parse_args()


def load_scores(path: Path, run_name: str | None = None) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    idx = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if run_name is not None and row.get("run_name") != run_name:
                continue
            score = row.get("score")
            if score is None:
                continue
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score_f):
                continue
            idx += 1
            rows.append((idx, score_f))
    return rows


def configure_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#D6D9DC", linewidth=0.8, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=9)


def main() -> int:
    args = parse_args()
    buffer_path = args.run_dir / "model_based_buffer.jsonl"
    if not buffer_path.exists():
        raise SystemExit(f"buffer not found: {buffer_path}")
    rows = load_scores(buffer_path, run_name=args.run_name)
    if not rows:
        raise SystemExit("no finite scores in buffer.")
    evaluations = [r[0] for r in rows]
    scores = [r[1] for r in rows]
    best: list[float] = []
    incumbent = math.inf
    for score in scores:
        incumbent = min(incumbent, score)
        best.append(incumbent)

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.scatter(evaluations, scores, s=14, color="#9AA0A6", alpha=0.5, label="Observed")
    axis.step(evaluations, best, where="post", linewidth=2.2, color=COLOR, label="Best so far")
    completed = max(r[0] for r in rows)
    total = args.total_iterations
    if total is not None and completed < total:
        title = f"NanoGPT LDM: LCB ({completed}/{total} interim)"
    else:
        title = "NanoGPT LDM: LCB"
    axis.set(
        title=title,
        xlabel="Real evaluation",
        ylabel="val_bpb (lower is better)",
    )
    configure_axis(axis)
    axis.legend(frameon=False)

    out = args.out or (args.run_dir / "plots" / "nanogpt_lcb_trajectory.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=220, bbox_inches="tight")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    best_score = min(scores)
    print(f"rows={len(rows)} best_val_bpb={best_score:.6f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Rebuild the TRAINING_PLAN main-results table from the files that exist right now.

The main objective is: does RL on top of SFT beat SFT-only, measured by Pareto
hypervolume at search budget 80, evaluated on G12D (in-distribution) and G12C (transfer).
Spearman(vina, proposal index) and the zero-variance rate are diagnostics from a side
investigation; neither is the main metric and neither substitutes for it.

Every cell is labelled with what kind of thing it is:
  DEMONSTRATED  a number produced by a real run, traceable to files on disk
  REPORTED-ONLY a claim that appears in prose with no artifact behind it
  CANCELLED     planned, started, then stopped
  NOT-DONE      never attempted
Failures and short runs stay in the denominators.
"""
import json, glob, re, subprocess, hashlib
from pathlib import Path

T = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl")
L = T.parent
OUT = {}

def sh(cmd, cwd=None):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                              cwd=cwd).stdout.strip()
    except Exception:
        return ""

# ---- 1. what the plan asks for -------------------------------------------------
PLAN = L/"_wt_handoff/rl/slime_launch/TRAINING_PLAN.md"
OUT["objective"] = {
    "source": str(PLAN),
    "question": "Does RL on top of SFT beat SFT-only?",
    "primary_metric": "Pareto front hypervolume at search budget 80",
    "eval_sets": {"G12D": "in-distribution", "G12C": "transfer (activity model differs; docking receptor 8UN5 shared)"},
    "runs_planned": {"R1": "base + acquisition-max", "R2": "SFT + acquisition-max (pivot)",
                     "R3": "SFT + real hypervolume delta", "R4": "SFT + acquisition-mean"},
    "seeds_planned": ">=3-5 per run",
    "note": "Spearman(vina, index) and zero_std are NOT this metric.",
}

# ---- 2. is the primary metric measured anywhere? --------------------------------
hv_hits = []
for pat in ("results/*.json", "results/*.csv", "results/**/*.json"):
    for f in glob.glob(str(T/pat), recursive=True):
        try:
            txt = Path(f).read_text(errors="replace")[:2_000_000]
        except Exception:
            continue
        if re.search(r'hypervolume|"hv"|pareto_front|\bHV\b', txt, re.I):
            hv_hits.append(f)
OUT["primary_metric_artifacts"] = {
    "files_mentioning_hv": sorted(hv_hits),
    "budget80_eval_outputs": [],   # filled below
}
b80 = [f for f in hv_hits if re.search(r'budget.?80|_80_', f)]
OUT["primary_metric_artifacts"]["budget80_eval_outputs"] = b80

# ---- 3. checkpoint census, by model scale --------------------------------------
def argv_of(d):
    lg = Path(d)/"train.log"
    if not lg.exists(): return ""
    return lg.read_text(errors="replace")[:400_000]

rows = []
for d in sorted(glob.glob(str(T/"runs/*/"))):
    d = Path(d)
    ck = d/"ckpt"
    iters = sorted(x for x in (p.name for p in ck.glob("iter_*"))) if ck.exists() else []
    head = argv_of(d)
    m_hidden = re.search(r'--hidden-size\s+(\d+)', head)
    m_layers = re.search(r'--num-layers\s+(\d+)', head)
    m_ns     = re.search(r'--n-samples-per-prompt\s+(\d+)', head)
    m_lr     = re.search(r'--lr\s+([0-9.eE+-]+)', head)
    m_seed   = re.search(r'--seed\s+(\d+)', head)
    m_load   = re.search(r'--(?:load|ref-load)\s+(\S+)', head)
    gp = d/"gp_history.jsonl"
    nprop = (sum(1 for _ in gp.open()) - 63) if gp.exists() else None
    rows.append(dict(
        run=d.name, n_ckpt=len(iters), last_iter=iters[-1] if iters else None,
        hidden=int(m_hidden.group(1)) if m_hidden else None,
        layers=int(m_layers.group(1)) if m_layers else None,
        n_samples=int(m_ns.group(1)) if m_ns else None,
        lr=m_lr.group(1) if m_lr else None,
        train_seed=int(m_seed.group(1)) if m_seed else None,
        init_from=m_load.group(1) if m_load else None,
        proposals=nprop,
    ))
OUT["runs"] = rows

by_scale = {}
for r in rows:
    key = f"hidden={r['hidden']}" if r["hidden"] else "unknown"
    s = by_scale.setdefault(key, dict(runs=0, with_ckpt=0))
    s["runs"] += 1
    s["with_ckpt"] += 1 if r["n_ckpt"] else 0
OUT["by_scale"] = by_scale

# ---- 4. the 9B question ---------------------------------------------------------
nine = [r for r in rows if (r["hidden"] or 0) >= 3584 or re.match(r'^(G\d|SN)', r["run"])]
OUT["nine_b"] = {
    "run_dirs": len(nine),
    "with_any_checkpoint": sum(1 for r in nine if r["n_ckpt"]),
    "verdict": None,   # set below
}
OUT["nine_b"]["verdict"] = (
    "NOT-DONE: no 9B run produced a checkpoint, so no 9B model exists to evaluate. "
    "Every checkpoint on disk belongs to the 1.5B pipeline pilot."
    if OUT["nine_b"]["with_any_checkpoint"] == 0 else "checkpoints exist; inspect")

# ---- 5. code / config identity ---------------------------------------------------
OUT["identity"] = {
    "handoff_head": sh(["git","rev-parse","HEAD"], cwd=str(L/"_wt_handoff")),
    "handoff_branch": sh(["git","rev-parse","--abbrev-ref","HEAD"], cwd=str(L/"_wt_handoff")),
    "runtime_head": sh(["git","rev-parse","HEAD"], cwd=str(L/"_wt_runtime")),
    "runtime_branch": sh(["git","rev-parse","--abbrev-ref","HEAD"], cwd=str(L/"_wt_runtime")),
}
for f in ("plan/harvest_lr0_arm.py", "plan/PRESPEC_lr0_control.md", "code/run_15b_param.sh",
          "launchers/run_train_real_slime_param.sh"):
    p = T/f
    if p.exists():
        OUT["identity"][f] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]

print(json.dumps(OUT, indent=1, ensure_ascii=False))

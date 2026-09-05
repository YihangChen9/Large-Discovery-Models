#!/bin/bash
# Reproducible CPU-only preparation for the hypervolume evaluation.
#
# Nothing here needs a GPU. It exists because the evaluation chain could not run at all
# before 2026-09-05: no single environment had rdkit AND sklearn/joblib AND gpytorch,
# and the saved QSAR models could not be unpickled from any working directory.
set -euo pipefail
T=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl
W=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/_wt_handoff
QSAR_SRC=$W/tasks/small_molecule/core/activity_modeling

echo "[1/3] packages (user site; the shared conda env is not modified)"
python3 -m pip install --user -q rdkit gpytorch botorch gauche

echo "[2/3] import check"
python3 - <<'PY'
import importlib, sys
need = ["rdkit","numpy","scipy","sklearn","joblib","yaml","pandas","torch","gpytorch","botorch","gauche"]
bad = []
for m in need:
    try: importlib.import_module(m)
    except Exception as e: bad.append(f"{m}: {type(e).__name__}")
if bad:
    sys.exit("MISSING: " + ", ".join(bad))
print("  all %d present" % len(need))
PY

echo "[3/3] QSAR models load and predict"
# The pickles were written by train_g12c_qsar.py running as __main__, so the pickle
# records the module name `train_g12c_qsar`. Loading therefore requires that script's
# directory on sys.path -- from anywhere else joblib raises
# ModuleNotFoundError: No module named 'train_g12c_qsar', which reads like a missing
# dependency but is a path problem.
QSAR_SRC="$QSAR_SRC" T="$T" python3 - <<'PY'
import os, sys, joblib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.environ["QSAR_SRC"])
T = os.environ["T"]
for name, d in [("G12C", f"{T}/results/g12c_qsar_20260901T010923Z"),
                ("G12D", f"{T}/results/g12d_qsar_matched_20260901T011826Z")]:
    m = joblib.load(f"{d}/best_model.joblib")
    y = m.predict(["CCO", "c1ccccc1C(=O)N"])
    print(f"  {name}: {type(m).__name__}  predict -> {y}")
PY

echo "CPU prerequisites OK"

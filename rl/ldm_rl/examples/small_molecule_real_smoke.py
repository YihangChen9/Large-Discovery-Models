"""Real-evaluation smoke test for the small-molecule env (CPU only).

Drives a few real rounds — real AutoDock Vina docking + activity NN + GP/SIR
selection — with a scripted proposer, and prints the per-round real metrics and
reward. This exercises the REAL path (not mock), including the acquisition and
hypervolume reward policies. No GPU needed.

Usage (inside the image / any env with the small-molecule real deps + vina):
    python small_molecule_real_smoke.py [rounds] [reward]
      rounds : number of rounds (default 3)
      reward : improvement | acquisition | hypervolume (default hypervolume)

Point VINA_BIN / NN_MODEL to real artifacts, or rely on the defaults below.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RL_ROOT = Path(__file__).resolve().parents[2]
for _p in (_RL_ROOT, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ldm_rl import EnvConfig  # noqa: E402
from ldm_rl.factories import build_env  # noqa: E402
from tasks.small_molecule.core.workflow import ExpandingMockCase2LLM  # noqa: E402


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    reward = sys.argv[2] if len(sys.argv) > 2 else "acquisition"
    # hypervolume needs a fixed oriented-space nadir (moving nadir disabled, PR #2)
    ref_point = (0.0, 5.0) if reward == "hypervolume" else None

    real_kwargs = dict(
        vina_bin=os.environ.get("VINA_BIN", "/mnt/data0/dock-project/bin/vina"),
        nn_model_path=os.environ.get(
            "NN_MODEL",
            str(_REPO_ROOT / "tasks/small_molecule/resources/models/best_g12d_model.joblib"),
        ),
        vina_pdb_id="8UN5",
        vina_chain_id="A",
        gp_device="cpu",
        vina_exhaustiveness=1,
        vina_n_poses=1,
        vina_max_workers=1,
    )
    print(f"[real-smoke] reward={reward} rounds={rounds} vina={real_kwargs['vina_bin']}", flush=True)

    env = build_env(
        "small_molecule",
        mode="real",
        config=EnvConfig(
            iterations=rounds,
            reservoir_size=2,
            evaluations_per_round=1,
            reward=reward,
            reward_ref_point=ref_point,
        ),
        **real_kwargs,
    )
    llm = ExpandingMockCase2LLM()

    obs = env.reset()
    total = 0.0
    for r in range(rounds):
        action = llm.chat("system", obs, json_mode=True)
        step = env.step(action)
        total += step.reward
        ev = step.info.get("evaluated") or []
        metrics = ev[0]["evaluation"]["metrics"] if ev else {}
        print(
            f"round={r} reward={step.reward:.6f} kind={step.info['reward_components'].get('kind')} "
            f"metrics={ {k: round(v, 4) for k, v in metrics.items()} } "
            f"done={step.done}",
            flush=True,
        )
        obs = step.observation
        if step.done:
            break
    print("SMOKE_OK " + json.dumps({"reward": reward, "rounds": rounds, "total_reward": round(total, 6)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Engine integration tests for the antibody LDMEngine campaign.

These cover the previously untested main-loop glue: warmup -> acquisition
switching, per-round decision records, and the legacy ``results.csv`` /
``llm_acq_decisions.jsonl`` exports, all driven through ``ldm_acq.run_one``
with a deterministic mock LLM and a random energy evaluator.
"""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ldm_tts.contracts import Candidate
from ldm_tts.optimization.records import BOObservation, SurrogateVector
from tasks.antibody.core import engine_adapters
from tasks.antibody.core.ldm_light import ldm_acq


def _args(tmp_path, **overrides) -> SimpleNamespace:
    defaults = dict(
        out_root=str(tmp_path / "runs"),
        seed=17,
        method="llm_gen",
        parallel_budget=8,
        n_evals=2,
        batch_size=1,
        gen_m=2,
        n_strategies=2,
        planner_mode="choices",
        softmax_eta=1.0,
        per_strategy_budget=0,
        pool_score="acq",
        selection_score="acq",
        bias_weight=0.05,
        acq="ei",
        acq_beta=1.0,
        acq_xi=0.001,
        n_init=2,
        include_antigen_context=False,
        max_retries=1,
        history_top_k=10,
        fallback_random=False,
        temperature=0.0,
        timeout_s=10,
        device="cpu",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _sequence(index: int) -> str:
    alphabet = "GPQST"
    chars = []
    for _ in range(11):
        chars.append(alphabet[index % len(alphabet)])
        index //= len(alphabet)
    return "".join(chars)


class DirectMockLLM:
    """Deterministic mock that emits developable direct sequences."""

    def __init__(self) -> None:
        self.calls = 0

    def call(self, prompt: str, temperature: float = 0.0, timeout_s: int = 30) -> str:
        self.calls += 1
        if '"task": "direct_cdrh3_generation"' in prompt:
            count = int(json.loads(prompt)["constraints"]["num_sequences"])
            return json.dumps([_sequence(self.calls + i) for i in range(count)])
        return json.dumps({
            "rationale": "deterministic mock policy",
            "trust_region": "LatinHyperCubeSampling(num=8)",
        })

    def call_many(
        self,
        prompt: str,
        temperature: float = 0.0,
        timeout_s: int = 30,
        n: int = 1,
    ) -> list[str]:
        return [self.call(prompt, temperature, timeout_s) for _ in range(int(n))]

    def close(self) -> None:
        return None


class RandomEnergyEvaluator:
    def energy(self, x) -> tuple[np.ndarray, list[str]]:
        sequences = ldm_acq.indices_to_seqs(x)
        rng = np.random.default_rng(len(sequences))
        return rng.random(len(sequences)), sequences


def _run_one(tmp_path, monkeypatch, args) -> "PathLike":
    monkeypatch.setattr(ldm_acq, "make_llm_client", lambda: DirectMockLLM())
    monkeypatch.setattr(
        ldm_acq,
        "make_evaluator",
        lambda _config, _antigen, _run_id: (
            RandomEnergyEvaluator(),
            {"tool": "random"},
        ),
    )
    return ldm_acq.run_one(
        {"seq_len": 11, "bbox": {"tool": "random"}},
        "TEST_ANTIGEN",
        17,
        args,
    )


def test_llm_gen_engine_artifacts(tmp_path, monkeypatch):
    run_dir = _run_one(tmp_path, monkeypatch, _args(tmp_path, method="llm_gen"))
    for name in (
        "campaign.json",
        "events.jsonl",
        "checkpoint.json",
        "summary.json",
        "budget.json",
        "status.json",
        "ldm_task_spec.json",
    ):
        assert (run_dir / name).exists(), name
    assert (run_dir / "results.csv").exists()
    assert (run_dir / "llm_acq_decisions.jsonl").exists()


def test_llm_gen_results_and_decisions(tmp_path, monkeypatch):
    run_dir = _run_one(
        tmp_path, monkeypatch, _args(tmp_path, method="llm_gen", n_evals=3)
    )
    lines = (run_dir / "results.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4  # header + 3 evaluations
    decisions = [
        json.loads(line)
        for line in (run_dir / "llm_acq_decisions.jsonl").read_text().splitlines()
    ]
    assert len(decisions) == 3
    assert all(decision["acquisition"]["used"] is False for decision in decisions)
    assert [decision["eval_start"] for decision in decisions] == [0, 1, 2]


def test_direct_max_uses_acquisition_only_after_warmup(tmp_path, monkeypatch):
    run_dir = _run_one(
        tmp_path,
        monkeypatch,
        _args(tmp_path, method="direct_max", n_evals=4, n_init=2, gen_m=3),
    )
    decisions = [
        json.loads(line)
        for line in (run_dir / "llm_acq_decisions.jsonl").read_text().splitlines()
    ]
    assert len(decisions) == 4
    assert [decision["acquisition"]["used"] for decision in decisions] == [
        False,
        False,
        True,
        True,
    ]
    assert decisions[2]["acquisition"]["selected_candidates"]


def test_direct_max_results_carry_acquisition_scores(tmp_path, monkeypatch):
    run_dir = _run_one(
        tmp_path,
        monkeypatch,
        _args(tmp_path, method="direct_max", n_evals=4, n_init=2, gen_m=3),
    )
    with (run_dir / "results.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert all(row["AcquisitionScore"] == "" for row in rows[:2])
    assert any(row["AcquisitionScore"] for row in rows[2:])


def test_engine_events_record_full_round_flow(tmp_path, monkeypatch):
    run_dir = _run_one(
        tmp_path, monkeypatch, _args(tmp_path, method="llm_gen", n_evals=2)
    )
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]
    event_types = [event["event_type"] for event in events]
    assert event_types.count("reservoir_expanded") == 2
    assert event_types.count("candidate_evaluated") == 2
    assert "campaign_finished" in event_types


def test_engine_respects_partial_final_evaluation_batch(tmp_path, monkeypatch):
    run_dir = _run_one(
        tmp_path,
        monkeypatch,
        _args(
            tmp_path,
            method="llm_gen",
            n_evals=3,
            batch_size=2,
            fallback_random=True,
        ),
    )

    with (run_dir / "results.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["observation_count"] == 3
    assert summary["stop_reason"] == "observation_target"


def test_policy_reservoir_scores_are_selected_by_engine_selector(tmp_path):
    args = _args(tmp_path, method="policy_max", n_init=2)
    selector = engine_adapters.AntibodyGPSelector(
        args=args,
        method_spec={"reduction": "max"},
        rng=np.random.default_rng(0),
    )
    selector.fit([
        BOObservation(
            "seen-a",
            (1.0,),
            SurrogateVector((0.0,), "test", metadata={"sequence": _sequence(1)}),
        ),
        BOObservation(
            "seen-b",
            (0.5,),
            SurrogateVector((1.0,), "test", metadata={"sequence": _sequence(2)}),
        ),
    ])
    candidates = [
        Candidate(
            "low",
            {"sequence": _sequence(3), "acquisition_score": 0.1, "mu": 0.0, "sigma": 1.0},
            _sequence(3),
        ),
        Candidate(
            "high",
            {"sequence": _sequence(4), "acquisition_score": 0.9, "mu": 0.0, "sigma": 1.0},
            _sequence(4),
        ),
    ]

    result = selector.select(candidates, {}, count=1)

    assert result.selected_candidate_ids == ("high",)
    assert result.metadata["mode"] == "precomputed_max"


def test_policy_workflow_passes_full_reservoir_to_engine_selector(
    tmp_path, monkeypatch
):
    from tasks.antibody.core.ldm_light import reservoir

    low = _sequence(20)
    high = _sequence(21)

    def fake_policy_reservoir(**_kwargs):
        representatives = [
            {
                "sequence": low,
                "strategy_index": 0,
                "ei": 0.1,
                "bias+ei": 0.1,
                "mu": 0.0,
                "sigma": 1.0,
            },
            {
                "sequence": high,
                "strategy_index": 1,
                "ei": 0.9,
                "bias+ei": 0.9,
                "mu": 0.0,
                "sigma": 1.0,
            },
        ]
        return [], {
            "source": "policy_max",
            "representatives": representatives,
            "selected_indices": [],
            "selection_probabilities": [],
        }

    monkeypatch.setattr(
        reservoir,
        "select_with_policy_reservoir",
        fake_policy_reservoir,
    )
    run_dir = _run_one(
        tmp_path,
        monkeypatch,
        _args(tmp_path, method="policy_max", n_evals=3, n_init=2),
    )

    with (run_dir / "results.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]["LastProtein"] == high
    assert float(rows[-1]["AcquisitionScore"]) == pytest.approx(0.9)
    events = [
        json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]
    policy_selection = [
        event
        for event in events
        if event["event_type"] == "candidates_selected"
    ][-1]
    assert policy_selection["payload"]["metadata"]["mode"] == "precomputed_max"

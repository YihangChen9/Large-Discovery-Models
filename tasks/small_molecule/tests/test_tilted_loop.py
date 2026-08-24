"""Engine integration tests for the small-molecule tilted case2 campaign.

The task's main loop now runs through ``ldm_tts.engine.LDMEngine`` with the
adapters in ``tasks.small_molecule.core.engine_adapters``. These tests drive
the same assembly as ``core.workflow.main`` and assert both the shared engine
artifacts and the legacy trajectory exports.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from ldm_tts.contracts import Candidate, EvaluationResult, Observation
from ldm_tts.engine import LDMEngine, LDMEngineConfig, LDMEngineState
from ldm_tts.engine.run_store import CampaignRuntime

from tasks.small_molecule.core import engine_adapters
from tasks.small_molecule.core.gp import GPConfig
import tasks.small_molecule.core.ldm_tilted_case2.loop as loop_mod
from tasks.small_molecule.core.ldm_tilted_case2.candidate_record import (
    CandidateRecord,
)
from tasks.small_molecule.core.ldm_tilted_case2.config import TiltedLDMCase2Config
from tasks.small_molecule.core.ldm_tilted_case2.loop import _score_smiles
from tasks.small_molecule.core.llm_advisor.client import MockLLMClient
from tasks.small_molecule.core.rng import RNG


def test_config_validation():
    TiltedLDMCase2Config(method="m1_direct_llm_sir")

    with pytest.raises(ValueError, match="method"):
        TiltedLDMCase2Config(method="unknown")
    with pytest.raises(ValueError, match="batch_size"):
        TiltedLDMCase2Config(method="m1_direct_llm_sir", batch_size=0)
    with pytest.raises(ValueError, match="budget"):
        TiltedLDMCase2Config(method="m1_direct_llm_sir", init_size=5, budget=4)
    with pytest.raises(ValueError, match="eta"):
        TiltedLDMCase2Config(method="m1_direct_llm_sir", eta_ehvi_tilt=-1.0)
    with pytest.raises(ValueError, match="alpha"):
        TiltedLDMCase2Config(method="m1_direct_llm_sir", alpha_base_measure=-1.0)
    with pytest.raises(ValueError, match="minimize"):
        TiltedLDMCase2Config(method="m1_direct_llm_sir", minimize=(True,))
    with pytest.raises(ValueError, match="ref_point"):
        TiltedLDMCase2Config(method="m1_direct_llm_sir", ref_point=(0.0,))
    with pytest.raises(ValueError, match="max_empty_reservoir_rounds"):
        TiltedLDMCase2Config(method="m1_direct_llm_sir", max_empty_reservoir_rounds=0)


def mock_scorer_vina(smiles_list):
    return [-float(len(smiles)) / 10.0 for smiles in smiles_list]


def mock_scorer_activity(smiles_list):
    return [5.0 + float(smiles.count("N")) for smiles in smiles_list]


def mock_analog_fn(seeds):
    out = []
    for seed in seeds:
        out.extend([seed + "C", seed + "N"])
    return out


def m1_llm():
    return MockLLMClient(scripted_responses=[
        json.dumps({"direct_smiles": [{"smiles": "CCCC", "rationale": "x"}, {"smiles": "CCCN", "rationale": "y"}]}),
        json.dumps({"direct_smiles": [{"smiles": "CCCCN", "rationale": "x"}, {"smiles": "CCCCO", "rationale": "y"}]}),
        json.dumps({"direct_smiles": [{"smiles": "CCNCC", "rationale": "x"}, {"smiles": "CCOCC", "rationale": "y"}]}),
    ])


def spec_for(cfg: TiltedLDMCase2Config):
    from tasks.small_molecule.core import workflow

    args = SimpleNamespace(
        method=cfg.method,
        kernel="sk" if cfg.gp_config.impl == "smiles-strkernel" else "fp",
        gp_fp_n_bits=int(cfg.gp_config.fp_n_bits),
        acq=cfg.acquisition,
        acq_weights="0.5,0.5",
        alpha=float(cfg.alpha_base_measure),
        eta=float(cfg.eta_ehvi_tilt),
        ehvi_n_samples=int(cfg.ehvi_n_samples),
        batch_size=int(cfg.batch_size),
        smiles_max_len=int(cfg.smiles_max_len),
        max_candidates_per_round=int(cfg.max_candidates_per_round),
        init_strategy=cfg.init_strategy,
        budget=int(cfg.budget),
        init_size=int(cfg.init_size),
    )
    return workflow.describe_ldm_task(args)


def run_campaign(
    cfg: TiltedLDMCase2Config,
    llm,
    tmp_path: Path,
    *,
    seeds=(),
    vina=mock_scorer_vina,
    activity=mock_scorer_activity,
    analog=None,
    resume: bool = False,
):
    """Assemble and run one engine campaign the way ``workflow.main`` does."""
    run_dir = Path(tmp_path)
    runtime = CampaignRuntime.open(
        run_dir,
        task="small_molecule",
        task_spec=spec_for(cfg),
        budget_limits=None,
        resume=resume,
    )
    evaluator = engine_adapters.SmilesCandidateEvaluator(vina, activity)
    state = LDMEngineState()
    if not resume and cfg.init_strategy == "seed_smiles":
        state = LDMEngineState(observations=_seed_observations(
            cfg, evaluator, runtime, seeds
        ))
    if resume:
        checkpoint = runtime.load_checkpoint()
        if checkpoint is not None:
            state = LDMEngineState.from_checkpoint(checkpoint)

    remaining = max(0, -(-(cfg.budget - len(state.observations)) // cfg.batch_size))
    iterations = state.next_round + remaining
    runtime.budget.limits = {
        "outer_iterations": iterations,
        "valid_search_candidates": iterations * cfg.max_candidates_per_round,
        "selected_candidates": len(state.observations) + iterations * cfg.batch_size,
        "external_evaluations": len(state.observations) + iterations * cfg.batch_size,
        "expensive_evaluation_attempts": len(state.observations) + iterations * cfg.batch_size,
        "successful_evaluations": len(state.observations) + iterations * cfg.batch_size,
        "benchmark_jobs": len(state.observations) + iterations * cfg.batch_size,
    }
    runtime.budget.write()

    domain = engine_adapters.SmilesCandidateDomain(cfg)
    expander = engine_adapters.SmilesReservoirExpander(
        cfg, llm, analog or mock_analog_fn, budget_hook=runtime.consume
    )
    encoder = None
    selector = None
    if cfg.method not in engine_adapters.DIRECT_ONLY_METHODS:
        encoder = engine_adapters.SmilesSurrogateEncoder(cfg.gp_config)
        selector = engine_adapters.TiltedAcquisitionSelector(cfg)
    engine = LDMEngine(
        task_spec=spec_for(cfg),
        expander=expander,
        candidate_domain=domain,
        evaluator=evaluator,
        runtime=runtime,
        surrogate_encoder=encoder,
        selector=selector,
    )
    result = engine.run(
        LDMEngineConfig(
            iterations=iterations,
            reservoir_size=cfg.max_candidates_per_round,
            evaluations_per_round=cfg.batch_size,
            max_empty_reservoir_rounds=(
                cfg.max_empty_reservoir_rounds if cfg.allow_early_stop else max(iterations, 1)
            ),
        ),
        state=state,
    )
    legacy_summary = engine_adapters.materialize_legacy_trajectory(runtime, result, cfg)
    return result, legacy_summary, runtime


def _seed_observations(cfg, evaluator, runtime, seeds):
    from tasks.small_molecule.core.ldm_tilted_case2.canonicalize import canonicalize_smiles

    canonical = []
    seen = set()
    for smiles in seeds:
        canon = canonicalize_smiles(smiles)
        if canon and canon not in seen:
            canonical.append(canon)
            seen.add(canon)
        if len(canonical) >= cfg.init_size:
            break
    observations = []
    for smiles in canonical:
        candidate = Candidate(
            candidate_id="mol-" + __import__("hashlib").sha256(smiles.encode()).hexdigest()[:12],
            payload={"smiles": smiles, "rationale": ""},
            canonical_key=smiles,
            source="seed_smiles",
        )
        runtime.consume_many({
            "external_evaluations": 1,
            "expensive_evaluation_attempts": 1,
            "selected_candidates": 1,
        })
        evaluation = evaluator.evaluate(candidate)
        if evaluation.succeeded:
            runtime.consume("successful_evaluations")
            runtime.consume("benchmark_jobs", int(evaluation.resource_usage.get("benchmark_jobs", 0)))
        observations.append(Observation(candidate=candidate, evaluation=evaluation))
        runtime.record("candidate_evaluated", observations[-1].to_dict(), candidate_id=candidate.candidate_id)
    return observations


def history_rows(result):
    return [
        (observation.candidate.payload["smiles"], (
            observation.evaluation.metrics.get("vina"),
            observation.evaluation.metrics.get("activity"),
        ))
        for observation in result.state.observations
    ]


def run_method(method, llm, tmp_path, **kwargs):
    cfg = TiltedLDMCase2Config(
        method=method,
        init_size=3,
        budget=6,
        m1_k_direct_llm=2,
        gp_config=GPConfig(device="cpu", fit_n_itersteps=2, fp_n_bits=128),
        **kwargs,
    )
    return run_campaign(cfg, llm, tmp_path, seeds=("CCO", "CCN", "CCC"))


def test_engine_runs_m1_mock_two_objectives(tmp_path):
    result, _summary, _runtime = run_method("m1_direct_llm_sir", m1_llm(), tmp_path)
    history = history_rows(result)
    assert len(history) == 6
    assert len(history[0][1]) == 2


def test_engine_does_not_collapse_multi_objective_scores(tmp_path):
    result, _summary, _runtime = run_method("m1_direct_llm_sir", m1_llm(), tmp_path)
    history = history_rows(result)
    assert all(isinstance(scores, tuple) and len(scores) == 2 for _smiles, scores in history)


def test_engine_eta_zero_base_only(tmp_path):
    result, _summary, _runtime = run_method("m1_direct_llm_sir", m1_llm(), tmp_path, eta_ehvi_tilt=0.0)
    assert len(history_rows(result)) == 6


def test_engine_alpha_zero_ehvi_only(tmp_path):
    result, _summary, _runtime = run_method("m1_direct_llm_sir", m1_llm(), tmp_path, alpha_base_measure=0.0)
    assert len(history_rows(result)) == 6


def test_engine_cold_start_does_not_score_seed_smiles(tmp_path):
    client = MockLLMClient(scripted_responses=[
        json.dumps({"direct_smiles": [{"smiles": "CCCC", "rationale": "cold"}]}),
        json.dumps({"direct_smiles": [{"smiles": "CCCCN", "rationale": "cold"}]}),
        json.dumps({"direct_smiles": [{"smiles": "CCCCO", "rationale": "cold"}]}),
    ])
    scored_batches = []

    def vina(smiles_list):
        batch = list(smiles_list)
        scored_batches.append(batch)
        return [-float(len(smiles)) / 10.0 for smiles in batch]

    cfg = TiltedLDMCase2Config(
        "m1_direct_llm_sir",
        init_size=3,
        init_strategy="llm_cold_start",
        budget=3,
        batch_size=1,
        m1_k_direct_llm=1,
        gp_config=GPConfig(device="cpu", fit_n_itersteps=2, fp_n_bits=128),
    )
    result, summary, _runtime = run_campaign(
        cfg, client, tmp_path, seeds=("CCO", "CCN", "CCC"), vina=vina
    )

    assert [smiles for smiles, _scores in history_rows(result)] == ["CCCC", "CCCCN", "CCCCO"]
    assert all(set(batch) & {"CCO", "CCN", "CCC"} == set() for batch in scored_batches)
    assert summary["history_size"] == 3


def test_engine_seed_initialization_strategy_preserves_seed_history(tmp_path):
    cfg = TiltedLDMCase2Config(
        "m1_direct_llm_sir",
        init_size=2,
        init_strategy="seed_smiles",
        budget=2,
        gp_config=GPConfig(device="cpu", fit_n_itersteps=2, fp_n_bits=128),
    )
    result, _summary, _runtime = run_campaign(cfg, m1_llm(), tmp_path, seeds=("CCO", "CCN", "CCC"))

    assert [smiles for smiles, _scores in history_rows(result)] == ["CCO", "CCN"]


def test_engine_resume_from_checkpoint_continues_cold_start(tmp_path):
    first_client = MockLLMClient(scripted_responses=[
        json.dumps({"direct_smiles": [{"smiles": "CCCC", "rationale": "first"}]}),
        json.dumps({"direct_smiles": [{"smiles": "CCCCN", "rationale": "second"}]}),
    ])
    first_cfg = TiltedLDMCase2Config(
        "m1_llm_one_step",
        init_size=1,
        init_strategy="llm_cold_start",
        budget=2,
        batch_size=1,
        m1_k_direct_llm=1,
    )
    first_result, _summary, _runtime = run_campaign(first_cfg, first_client, tmp_path)
    assert [smiles for smiles, _scores in history_rows(first_result)] == ["CCCC", "CCCCN"]

    resume_client = MockLLMClient(scripted_responses=[
        json.dumps({"direct_smiles": [{"smiles": "CCCCO", "rationale": "third"}]}),
        json.dumps({"direct_smiles": [{"smiles": "CCNCC", "rationale": "fourth"}]}),
    ])
    resume_cfg = TiltedLDMCase2Config(
        "m1_llm_one_step",
        init_size=1,
        init_strategy="llm_cold_start",
        budget=4,
        batch_size=1,
        m1_k_direct_llm=1,
    )
    resumed_result, summary, runtime = run_campaign(
        resume_cfg, resume_client, tmp_path, resume=True
    )

    assert [smiles for smiles, _scores in history_rows(resumed_result)] == [
        "CCCC",
        "CCCCN",
        "CCCCO",
        "CCNCC",
    ]
    rounds = [json.loads(line) for line in (tmp_path / "rounds.jsonl").read_text().splitlines()]
    assert [record["round_idx"] for record in rounds] == [0, 1, 2, 3]
    assert summary["history_size"] == 4
    assert summary["llm_call_count"] == 4
    assert runtime.run_dir.joinpath("checkpoint.json").exists()


def test_fresh_run_uses_a_fresh_campaign_dir(tmp_path):
    stale_dir = tmp_path / "stale"
    stale_dir.mkdir()
    (stale_dir / "rounds.jsonl").write_text(
        json.dumps({"round_idx": 999, "stale": True}) + "\n",
        encoding="utf-8",
    )

    cfg = TiltedLDMCase2Config(
        "m1_direct_llm_sir",
        init_size=3,
        budget=6,
        m1_k_direct_llm=2,
        gp_config=GPConfig(device="cpu", fit_n_itersteps=2, fp_n_bits=128),
    )
    runtime = CampaignRuntime.open(
        stale_dir,
        task="small_molecule",
        task_spec=spec_for(cfg),
        budget_limits=None,
        resume=False,
    )
    assert (stale_dir / "campaign.json").exists()
    # The legacy rounds file stays untouched; engine artifacts are new.
    rounds = [json.loads(line) for line in (stale_dir / "rounds.jsonl").read_text().splitlines()]
    assert rounds[0]["stale"] is True
    assert (stale_dir / "events.jsonl").exists()


def test_engine_retries_transient_empty_reservoir(tmp_path):
    llm = MockLLMClient(scripted_responses=[
        json.dumps({"direct_smiles": []}),
        json.dumps({"direct_smiles": [{"smiles": "CCCC", "rationale": "x"}]}),
        json.dumps({"direct_smiles": [{"smiles": "CCCCN", "rationale": "y"}]}),
    ])
    cfg = TiltedLDMCase2Config(
        "m1_direct_llm_sir",
        init_size=3,
        budget=5,
        m1_k_direct_llm=1,
        gp_config=GPConfig(device="cpu", fit_n_itersteps=2, fp_n_bits=128),
    )
    result, summary, _runtime = run_campaign(cfg, llm, tmp_path, seeds=("CCO", "CCN", "CCC"))
    assert len(history_rows(result)) == 5
    assert summary["early_stop_reason"] == "iteration_budget"


def test_engine_stops_after_empty_reservoir_limit(tmp_path):
    llm = MockLLMClient(
        scripted_responses=[json.dumps({"direct_smiles": []})] * 24
    )
    cfg = TiltedLDMCase2Config(
        "m1_direct_llm_sir",
        init_size=3,
        budget=5,
        m1_k_direct_llm=1,
        max_empty_reservoir_rounds=2,
        gp_config=GPConfig(device="cpu", fit_n_itersteps=2, fp_n_bits=128),
    )
    result, summary, _runtime = run_campaign(cfg, llm, tmp_path, seeds=("CCO", "CCN", "CCC"))
    assert len(history_rows(result)) == 3
    assert summary["early_stop_reason"] == "empty_reservoir_limit"


def test_trace_jsonl_contains_required_fields(tmp_path):
    _result, _summary, _runtime = run_method("m1_direct_llm_sir", m1_llm(), tmp_path)
    line = (tmp_path / "rounds.jsonl").read_text().splitlines()[0]
    record = json.loads(line)
    assert "q0_entropy" in record
    assert "prob_effective_sample_size" in record
    assert "candidates" in record


def test_trace_selection_results_have_probabilities_and_scores(tmp_path):
    _result, _summary, _runtime = run_method("m1_direct_llm_sir", m1_llm(), tmp_path)
    rounds = [json.loads(line) for line in (tmp_path / "rounds.jsonl").read_text().splitlines()]
    for record in rounds:
        selection = record["selection_results"]
        assert selection["selected_smiles"]
        assert selection["selected_scores"]
        assert len(selection["selected_probabilities"]) == len(selection["selected_smiles"])


def test_trace_round_records_raw_llm_inputs_outputs_and_results(tmp_path):
    _result, _summary, _runtime = run_method("m1_direct_llm_sir", m1_llm(), tmp_path)
    first = json.loads((tmp_path / "rounds.jsonl").read_text().splitlines()[0])

    attempt = first["llm_attempts"][0]
    assert attempt["system_prompt"]
    assert "Generate up to" in attempt["user_prompt"]
    assert attempt["raw_output"] == attempt["raw_text"]
    assert attempt["parsed_json"]["direct_smiles"]

    assert first["selection_results"]["selected_smiles"]
    assert first["selection_results"]["selected_scores"]
    assert first["selection_results"]["ehvi_fallback_reason"] is None


def test_trace_summary_counts_llm_calls(tmp_path):
    _result, _summary, _runtime = run_method("m1_direct_llm_sir", m1_llm(), tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["llm_call_count"] > 0
    assert summary["final_hypervolume"] is not None


def test_engine_artifacts_exist_for_mock_campaign(tmp_path):
    _result, _summary, _runtime = run_method("m1_direct_llm_sir", m1_llm(), tmp_path)
    for name in ("events.jsonl", "checkpoint.json", "summary.json", "campaign.json",
                 "budget.json", "status.json", "ldm_task_spec.json"):
        assert (tmp_path / name).exists(), name


def test_workflow_entrypoint_uses_shared_campaign_algorithm(tmp_path):
    from tasks.small_molecule.core import workflow

    run_dir = tmp_path / "workflow-campaign"
    rc = workflow.main([
        "--mock",
        "--method",
        "m1_llm_one_step",
        "--budget",
        "2",
        "--batch-size",
        "1",
        "--init-size",
        "1",
        "--init-strategy",
        "llm_cold_start",
        "--m1-k-direct-llm",
        "1",
        "--output-dir",
        str(run_dir),
    ])

    assert rc == 0
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["successful_evaluation_count"] == 2
    assert summary["stop_reason"] == "successful_evaluation_target"


def test_workflow_seed_initialization_is_engine_owned(tmp_path):
    from tasks.small_molecule.core import workflow

    run_dir = tmp_path / "seeded-workflow"
    assert workflow.main([
        "--mock",
        "--method",
        "m1_llm_one_step",
        "--budget",
        "2",
        "--batch-size",
        "1",
        "--init-size",
        "2",
        "--init-strategy",
        "seed_smiles",
        "--seed-smiles",
        "CCO,CCN",
        "--output-dir",
        str(run_dir),
    ]) == 0

    events = [
        json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]
    expansions = [event for event in events if event["event_type"] == "reservoir_expanded"]
    evaluations = [event for event in events if event["event_type"] == "candidate_evaluated"]
    assert len(expansions) == 2
    assert all(event["payload"]["metadata"]["phase"] == "initialization" for event in expansions)
    assert len(evaluations) == 2
    assert all(
        event["payload"]["candidate"]["source"] == "seed_smiles"
        for event in evaluations
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["successful_evaluation_count"] == 2
    assert summary["round_count"] == 0


def test_score_smiles_retries_transient_nonfinite_values():
    class TransientVinaFailure:
        def __init__(self):
            self.calls = []

        def __call__(self, smiles_list):
            smiles = list(smiles_list)
            self.calls.append(smiles)
            if smiles == ["CCO", "CCN"]:
                return [float("nan"), -4.2]
            if smiles == ["CCO"]:
                return [-3.7]
            return [-4.2]

    vina = TransientVinaFailure()

    scores = _score_smiles(
        ["CCO", "CCN"],
        (vina, lambda smiles_list: [5.1, 5.2]),
    )

    assert scores == [(-3.7, 5.1), (-4.2, 5.2)]
    assert vina.calls == [["CCO", "CCN"], ["CCO"]]


def test_score_smiles_raises_scorer_exceptions():
    def broken_scorer(_smiles_list):
        raise RuntimeError("receptor prep failed")

    with pytest.raises(RuntimeError, match="broken_scorer failed.*receptor prep failed"):
        _score_smiles(["CCO"], (broken_scorer, lambda smiles_list: [5.1]))


def test_engine_one_step_records_failed_observation_for_unscorable_candidate(tmp_path):
    client = MockLLMClient(scripted_responses=[
        json.dumps({"direct_smiles": [{"smiles": "CCN", "rationale": "docking fail"}]}),
        json.dumps({"direct_smiles": [{"smiles": "CCC", "rationale": "scorable"}]}),
    ])

    def vina(smiles_list):
        return [float("nan") if smiles == "CCN" else -3.3 for smiles in smiles_list]

    cfg = TiltedLDMCase2Config(
        "m1_llm_one_step",
        init_size=1,
        budget=3,
        batch_size=1,
        m1_k_direct_llm=1,
    )
    result, summary, _runtime = run_campaign(
        cfg,
        client,
        tmp_path,
        seeds=("CCO",),
        vina=vina,
        activity=lambda smiles: [6.0 for _ in smiles],
    )

    history = history_rows(result)
    assert [smiles for smiles, _scores in history] == ["CCO", "CCN", "CCC"]
    assert history[1][1] == (None, 6.0)
    failed = [obs for obs in result.state.observations if not obs.evaluation.succeeded]
    assert len(failed) == 1
    assert summary["history_size"] == 3
    rounds = [json.loads(line) for line in (tmp_path / "rounds.jsonl").read_text().splitlines()]
    assert rounds[0]["selection_results"]["failed_evaluations"][0]["smiles"] == "CCN"


def test_tilted_selector_samples_by_tilted_probability(monkeypatch):
    cfg = TiltedLDMCase2Config(
        "m1_stratified_direct_llm_oversample_sir",
        batch_size=1,
        gp_config=GPConfig(device="cpu", fit_n_itersteps=2, fp_n_bits=128),
    )
    first = Candidate(
        "mol-a", {"smiles": "CCC"}, "CCC", source="s1",
        metadata={"occurrence_by_source": {"s1": 3}, "q0_base_mass": 0.75},
    )
    second = Candidate(
        "mol-b", {"smiles": "CCN"}, "CCN", source="s1",
        metadata={"occurrence_by_source": {"s1": 1}, "q0_base_mass": 0.25},
    )

    def fake_ehvi(_history, candidates, _cfg, _rng):
        for candidate in candidates:
            candidate.ehvi = 0.0
        return SimpleNamespace(ehvi=np.array([0.0, 0.0]), fallback_reason=None)

    monkeypatch.setattr(engine_adapters, "compute_ehvi_for_candidates", fake_ehvi)
    monkeypatch.setattr(engine_adapters, "gumbel_top_k", lambda _prob, _k, _rng: [0])

    selector = engine_adapters.TiltedAcquisitionSelector(cfg, RNG(0))
    selector.history = [("CCO", (-3.0, 6.0))]
    selection = selector.select([first, second], {}, count=1)

    assert selection.selected_candidate_ids == ("mol-a",)
    assert selection.metadata["selection_mode"] == "ehvi_sir"
    assert selection.metadata["selected_probabilities"] == [pytest.approx(0.75)]
    assert selection.fallback_reason is None

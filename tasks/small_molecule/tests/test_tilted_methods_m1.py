import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tasks.small_molecule.core.ldm_tilted_case2.config import TiltedLDMCase2Config
from tasks.small_molecule.core.ldm_tilted_case2.methods.direct_llm import (
    LLM_DIRECT_CHUNK_SIZE,
    M1_MAX_PARALLEL_LLM_CALLS,
    M1_STRATEGIES,
    DirectLLMReservoirBuilder,
)
from tasks.small_molecule.core.llm_advisor.client import MockLLMClient
from tasks.small_molecule.core.rng import RNG


def build_client():
    return MockLLMClient(
        scripted_responses=[
            json.dumps(
                {
                    "direct_smiles": [
                        {"smiles": "CCO", "rationale": "first"},
                        {"smiles": "CCO", "rationale": "dup"},
                        {"smiles": "CCN", "rationale": "second"},
                    ]
                }
            )
        ]
    )


def test_m1_builds_candidates_from_llm_json():
    cfg = TiltedLDMCase2Config("m1_direct_llm_sir", m1_k_direct_llm=3)
    result = DirectLLMReservoirBuilder().build([], cfg, build_client(), RNG(0))
    assert [c.canonical_smiles for c in result.candidates] == ["CCO", "CCN"]
    assert result.llm_attempts


def test_m1_q0_uses_duplicate_counts():
    cfg = TiltedLDMCase2Config("m1_direct_llm_sir", m1_k_direct_llm=3)
    result = DirectLLMReservoirBuilder().build([], cfg, build_client(), RNG(0))
    masses = {c.canonical_smiles: c.q0_base_mass for c in result.candidates}
    assert masses["CCO"] == 2 / 3
    assert masses["CCN"] == 1 / 3


def test_m1_rationale_not_used_in_weights():
    cfg = TiltedLDMCase2Config("m1_direct_llm_sir", m1_k_direct_llm=3)
    result = DirectLLMReservoirBuilder().build([], cfg, build_client(), RNG(0))
    assert result.candidates[0].metadata["rationales"] == ["first", "dup"]
    assert result.candidates[0].q0_base_mass == 2 / 3


def test_m1_splits_large_direct_requests():
    requested = LLM_DIRECT_CHUNK_SIZE + 1
    client = MockLLMClient(
        scripted_responses=[
            json.dumps({"direct_smiles": [{"smiles": "CCO", "rationale": "batch 1"}]}),
            json.dumps({"direct_smiles": [{"smiles": "CCN", "rationale": "batch 2"}]}),
        ]
    )
    cfg = TiltedLDMCase2Config("m1_direct_llm_sir", m1_k_direct_llm=requested)

    result = DirectLLMReservoirBuilder().build([], cfg, client, RNG(0))

    assert len(client.call_log) == 2
    assert f"Generate up to {LLM_DIRECT_CHUNK_SIZE} valid, unique candidate SMILES." in client.call_log[0]["user"]
    assert "Generate up to 1 valid, unique candidate SMILES." in client.call_log[1]["user"]
    assert "mutation, crossover, and scaffold-hop" in client.call_log[0]["user"]
    assert [c.canonical_smiles for c in result.candidates] == ["CCO", "CCN"]


def test_m1_stratified_uses_strategy_prompts_and_sources():
    client = MockLLMClient(
        scripted_responses=[
            json.dumps({"direct_smiles": [{"smiles": "CCO", "rationale": "local"}]}),
            json.dumps({"direct_smiles": [{"smiles": "CCN", "rationale": "activity"}]}),
            json.dumps({"direct_smiles": [{"smiles": "CCC", "rationale": "dock"}]}),
            json.dumps({"direct_smiles": [{"smiles": "CCCl", "rationale": "diverse"}]}),
        ]
    )
    cfg = TiltedLDMCase2Config(
        "m1_stratified_direct_llm_sir",
        m1_k_direct_llm=4,
        m1_q0_smoothing=0.5,
    )

    result = DirectLLMReservoirBuilder().build([("CCCC", (-1.0, 5.1))], cfg, client, RNG(0))

    assert len(client.call_log) == len(M1_STRATEGIES)
    assert all("Generation focus:" in call["user"] for call in client.call_log)
    assert all("Strategy:" not in call["user"] for call in client.call_log)
    assert [source.metadata["strategy"] for source in result.sources] == list(M1_STRATEGIES)
    assert {c.canonical_smiles for c in result.candidates} == {"CCO", "CCN", "CCC", "CCCl"}


def test_m1_strategies_do_not_hard_code_structural_answers():
    joined = " ".join(M1_STRATEGIES).lower()

    for forbidden in ["halogenated", "ring-rich", "bridged", "compact cyclic", "linear polyamine"]:
        assert forbidden not in joined


def test_m1_llm_parallelism_uses_smaller_chunks_and_more_workers():
    assert LLM_DIRECT_CHUNK_SIZE == 8
    assert M1_MAX_PARALLEL_LLM_CALLS == 64


def test_m1_stratified_smoothing_is_applied():
    client = MockLLMClient(
        scripted_responses=[
            json.dumps({"direct_smiles": [{"smiles": "CCO", "rationale": "a"}]}),
            json.dumps({"direct_smiles": [{"smiles": "CCO", "rationale": "b"}]}),
            json.dumps({"direct_smiles": [{"smiles": "CCN", "rationale": "c"}]}),
            json.dumps({"direct_smiles": [{"smiles": "CCC", "rationale": "d"}]}),
        ]
    )
    cfg = TiltedLDMCase2Config(
        "m1_stratified_direct_llm_sir",
        m1_k_direct_llm=4,
        m1_q0_smoothing=1.0,
    )

    result = DirectLLMReservoirBuilder().build([], cfg, client, RNG(0))
    masses = {c.canonical_smiles: c.q0_base_mass for c in result.candidates}

    assert math.isclose(masses["CCO"], 3 / 7)
    assert math.isclose(masses["CCN"], 2 / 7)
    assert math.isclose(masses["CCC"], 2 / 7)


def test_m1_stratified_refills_when_filters_collapse_reservoir():
    client = MockLLMClient(
        scripted_responses=[
            json.dumps({"direct_smiles": [{"smiles": "CCO", "rationale": "evaluated"}]}),
            json.dumps({"direct_smiles": [{"smiles": "CCO", "rationale": "evaluated"}]}),
            json.dumps({"direct_smiles": [{"smiles": "CCO", "rationale": "evaluated"}]}),
            json.dumps({"direct_smiles": [{"smiles": "CCO", "rationale": "evaluated"}]}),
            json.dumps({
                "direct_smiles": [
                    {"smiles": "CCN", "rationale": "refill one"},
                    {"smiles": "CCC", "rationale": "refill two"},
                ]
            }),
        ]
    )
    cfg = TiltedLDMCase2Config(
        "m1_stratified_direct_llm_sir",
        m1_k_direct_llm=4,
        m1_q0_smoothing=0.5,
    )

    result = DirectLLMReservoirBuilder().build([("CCO", (-1.0, 6.0))], cfg, client, RNG(0))

    assert len(client.call_log) == len(M1_STRATEGIES) + 1
    assert "Round feedback:" in client.call_log[-1]["user"]
    assert result.parsed_llm_json["refill_rounds"] == 1
    assert {candidate.canonical_smiles for candidate in result.candidates} == {"CCN", "CCC"}
    assert result.drop_counts["evaluated"] == 4


def test_m1_stratified_llm_only_selects_llm_order_without_ehvi(monkeypatch, tmp_path):
    def fail_ehvi(*_args, **_kwargs):
        raise AssertionError("LLM-only method must not call BO/EHVI")

    monkeypatch.setattr(
        "tasks.small_molecule.core.engine_adapters.compute_ehvi_for_candidates", fail_ehvi
    )

    client = MockLLMClient(
        scripted_responses=[
            json.dumps({
                "direct_smiles": [
                    {"smiles": "CCN", "rationale": "first"},
                    {"smiles": "CCC", "rationale": "second"},
                ]
            })
        ]
    )
    cfg = TiltedLDMCase2Config(
        "m1_stratified_direct_llm_only",
        init_size=1,
        budget=2,
        batch_size=1,
        m1_k_direct_llm=1,
    )

    # Drive the engine assembly directly: the LLM-only method runs without a
    # selector, so the engine evaluates candidates in reservoir (LLM) order.
    from tasks.small_molecule.core import engine_adapters
    from ldm_tts.engine import LDMEngine, LDMEngineConfig, LDMEngineState
    from ldm_tts.engine.run_store import CampaignRuntime
    from tasks.small_molecule.tests.test_tilted_loop import run_campaign

    result, summary, _runtime = run_campaign(
        cfg,
        client,
        tmp_path,
        seeds=("CCO",),
        vina=lambda smiles: [-1.0 for _ in smiles],
        activity=lambda smiles: [6.0 for _ in smiles],
        analog=lambda _seeds: [],
    )

    history = [
        (observation.candidate.payload["smiles"], observation.evaluation.metrics)
        for observation in result.state.observations
    ]
    assert history[-1][0] == "CCN"
    assert summary["method"] == "m1_stratified_direct_llm_only"
    assert summary["history_size"] == 2


def test_m1_llm_one_step_requests_and_keeps_one_candidate():
    client = MockLLMClient(
        scripted_responses=[
            json.dumps({
                "direct_smiles": [
                    {"smiles": "CCN", "rationale": "first"},
                    {"smiles": "CCC", "rationale": "extra should be ignored"},
                ]
            })
        ]
    )
    cfg = TiltedLDMCase2Config("m1_llm_one_step", m1_k_direct_llm=128)

    result = DirectLLMReservoirBuilder().build([], cfg, client, RNG(0))

    assert len(client.call_log) == 1
    assert "Generate up to 1 valid, unique candidate SMILES." in client.call_log[0]["user"]
    assert [candidate.canonical_smiles for candidate in result.candidates] == ["CCN"]
    assert result.parsed_llm_json["direct_batches"][0]["requested_count"] == 1


def test_m1_llm_one_step_keeps_requesting_one_until_valid_candidate():
    client = MockLLMClient(
        scripted_responses=[
            json.dumps({"direct_smiles": [{"smiles": "CCO", "rationale": "already seen"}]}),
            json.dumps({"direct_smiles": [{"smiles": "CCO", "rationale": "still seen"}]}),
            json.dumps({"direct_smiles": [{"smiles": "CCO", "rationale": "still seen"}]}),
            json.dumps({"direct_smiles": [{"smiles": "CCN", "rationale": "new"}]}),
        ]
    )
    cfg = TiltedLDMCase2Config("m1_llm_one_step", m1_k_direct_llm=1)

    result = DirectLLMReservoirBuilder().build([("CCO", (-1.0, 6.0))], cfg, client, RNG(0))

    assert len(client.call_log) == 4
    assert all("Generate up to 1 valid, unique candidate SMILES." in call["user"] for call in client.call_log)
    assert [candidate.canonical_smiles for candidate in result.candidates] == ["CCN"]
    assert result.parsed_llm_json["refill_rounds"] == 3


def test_m1_oversample_method_uses_stratified_batches():
    client = MockLLMClient(
        scripted_responses=[
            json.dumps({"direct_smiles": [{"smiles": "CCO", "rationale": "a"}]}),
            json.dumps({"direct_smiles": [{"smiles": "CCN", "rationale": "b"}]}),
            json.dumps({"direct_smiles": [{"smiles": "CCC", "rationale": "c"}]}),
            json.dumps({"direct_smiles": [{"smiles": "CCCl", "rationale": "d"}]}),
        ]
    )
    cfg = TiltedLDMCase2Config(
        "m1_stratified_direct_llm_oversample_sir",
        m1_k_direct_llm=4,
        m1_q0_smoothing=0.5,
    )

    result = DirectLLMReservoirBuilder().build([], cfg, client, RNG(0))

    assert len(client.call_log) == len(M1_STRATEGIES)
    assert [source.metadata["strategy"] for source in result.sources] == list(M1_STRATEGIES)
    assert result.parsed_llm_json["requested_total"] == 4


def test_m1_oversample_refills_until_pool_target():
    client = MockLLMClient(
        scripted_responses=[
            json.dumps({"direct_smiles": [{"smiles": "CCO", "rationale": "a"}]}),
            json.dumps({"direct_smiles": [{"smiles": "CCO", "rationale": "b"}]}),
            json.dumps({"direct_smiles": [{"smiles": "CCN", "rationale": "c"}]}),
            json.dumps({"direct_smiles": [{"smiles": "CCN", "rationale": "d"}]}),
            json.dumps({
                "direct_smiles": [
                    {"smiles": "CCC", "rationale": "refill"},
                    {"smiles": "CCCl", "rationale": "refill"},
                ]
            }),
        ]
    )
    cfg = TiltedLDMCase2Config(
        "m1_stratified_direct_llm_oversample_sir",
        m1_k_direct_llm=4,
        max_candidates_per_round=3,
        m1_q0_smoothing=0.5,
    )

    result = DirectLLMReservoirBuilder().build([("CCCC", (-1.0, 5.1))], cfg, client, RNG(0))

    assert len(client.call_log) == len(M1_STRATEGIES) + 1
    assert result.parsed_llm_json["min_valid_candidates"] == 3
    assert "target_min_valid_candidates" in client.call_log[-1]["user"]
    assert len(result.candidates) >= 3


def test_m1_keeps_near_recent_candidates_for_ehvi_resampling():
    client = MockLLMClient(
        scripted_responses=[
            json.dumps({
                "direct_smiles": [
                    {"smiles": "CCCCNCCNCCCCCNCCNCCCC", "rationale": "near"},
                    {"smiles": "CCCCNCCNCCCCCNCCNCCCCC", "rationale": "near"},
                    {"smiles": "CCCCCCCCCCCCNCCNCCCCCNCCNCCCC", "rationale": "near"},
                    {"smiles": "c1ccccc1N", "rationale": "alternative"},
                    {"smiles": "CCOC(=O)N", "rationale": "alternative"},
                    {"smiles": "CCSCC", "rationale": "alternative"},
                    {"smiles": "CC(C)O", "rationale": "alternative"},
                    {"smiles": "C1CCNCC1", "rationale": "alternative"},
                ]
            })
        ]
    )
    history = [
        ("CCO", (-1.0, 5.1)),
        ("CCN", (-1.1, 5.2)),
        ("CCC", (-1.2, 5.3)),
        ("CCCN", (-1.3, 5.4)),
        ("CCCC", (-1.4, 5.5)),
        ("CCCCN", (-3.3, 5.65)),
        ("CCCCNCCN", (-4.4, 5.69)),
        ("CCCCNCCNCCCCCN", (-5.0, 5.85)),
    ]
    cfg = TiltedLDMCase2Config("m1_direct_llm_sir", m1_k_direct_llm=8)

    result = DirectLLMReservoirBuilder().build(history, cfg, client, RNG(0))

    assert "near_recent" not in result.drop_counts
    assert "CCCCNCCNCCCCCNCCNCCCC" in {
        candidate.canonical_smiles for candidate in result.candidates
    }
    assert len(result.candidates) == 8

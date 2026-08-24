# Unified LDM Data Collection

This document explains how to collect fine-tuning data across the three tasks
that currently emit fine-tuning data, using the shared `ldm-2.0` intermediate
representation:

- `nanogpt`: training-program search over `train.py` parameter edits.
- `small_molecule`: small-molecule design over SMILES candidates.
- `antibody`: CDRH3 sequence design for AntBO.

The goal is to collect the data the model actually needs for supervised
fine-tuning:

```text
model-visible task + search state + request  ->  accepted teacher action
```

The important boundary is the teacher-model proposal event. Collect the prompt
state before the call and the accepted action after parsing/validation. Keep BO
selection, scores, runtime metadata, and provenance for auditing, but do not let
that metadata leak into the model-visible instruction unless it was genuinely
visible to the teacher model at proposal time.

For the full schema, see [data/SCHEMA.md](../data/SCHEMA.md). For the collection,
augmentation, rendering, and validation workflow, see
[data/README.md](../data/README.md).

## Output Files

Runtime collection writes three files:

```text
ldm_ir.jsonl       # ldm-2.0 intermediate records, one per accepted proposal event
ldm_sft.jsonl      # rendered LlamaFactory Alpaca rows
dataset_info.json  # LlamaFactory dataset registration snippet
```

`ldm_ir.jsonl` is the durable source of truth. `ldm_sft.jsonl` is a rendered
training artifact that can be regenerated from the IR.

Each IR record may include an extra top-level `collection` object:

```json
{
  "collection": {
    "provenance": {"run_id": "...", "round_idx": 7},
    "outcome": {"selected": ["..."], "scores": [0.1]}
  }
}
```

`collection` is for auditing and filtering only. The renderer ignores it, so it
does not become part of the model instruction.

## Shared Data Interface

Collection, expert augmentation, and rendering are exposed together through
the [`ldm_tts.data`](../ldm_tts/data/__init__.py) package. Implementations live in
focused modules including `ldm_tts/data/collection.py`, `ldm_tts/data/ir.py`,
`ldm_tts/data/rendering.py`, and `ldm_tts/data/augmentation.py`; task code
should import the public data interface.

Use these helpers from task code:

```python
from ldm_tts.data import (
    DataCollectionSink,
    ExpertJustificationPipeline,
    make_complete_design_ir,
    make_parameter_edit_ir,
    smallmol_irs_from_round_record,
)
```

Main utilities:

- `DataCollectionSink.from_env(default_root=...)`: opt-in append-only writer.
- `make_complete_design_ir(...)`: builder for tasks whose action proposes
  complete objects, such as SMILES or CDRH3 sequences.
- `make_parameter_edit_ir(...)`: builder for tasks whose action edits a parent
  state or expands an active feature space, such as nanogpt operation search.
- `render_record(ir)`: render one IR record into Alpaca format.
- `validate_ir_record(ir)`: validate the minimum `ldm-2.0` contract.
- `ExpertJustificationPipeline`: add resumable expert reasoning to IR or Alpaca
  records through an injected model adapter.

Runtime collection is controlled with environment variables:

```bash
LDM_DATA_COLLECTION_ENABLED=1
LDM_DATA_COLLECTION_DIR=/path/to/collection
LDM_DATA_COLLECTION_RENDER=prose
LDM_DATA_COLLECTION_STRIP_PARENT_ARTIFACT=0
```

Meanings:

- `LDM_DATA_COLLECTION_ENABLED`: truthy value enables collection.
- `LDM_DATA_COLLECTION_DIR`: optional shared output directory. If omitted, each
  task hook may use a run-local default such as `<trajectory_dir>/ldm_data`.
- `LDM_DATA_COLLECTION_RENDER`: `prose` or `json`; `prose` is recommended for SFT.
- `LDM_DATA_COLLECTION_STRIP_PARENT_ARTIFACT`: when truthy, omit large parent
  artifacts such as `train.py` from rendered prompts. Use only when context
  length or memory forces it, because it can create train/inference mismatch.

## Expert Justification Augmentation

The preferred augmentation input is `ldm_ir.jsonl`. Expert reasoning is written
to `action.reasoning`, keeping the justification structurally attached to the
accepted action. Use `--sft-output` to regenerate Alpaca data from the augmented
IR in the same run:

```bash
export LLM_BASE_URL=https://your-model-host.example/v1
export LLM_MODEL_NAME=your-served-model
export LLM_API_KEY=your-secret

python data/augment.py \
  --input data/generated/my_campaign/ldm_ir.jsonl \
  --output data/generated/my_campaign/ldm_ir_augmented.jsonl \
  --checkpoint data/generated/my_campaign/augmentation.checkpoint.jsonl \
  --sft-output data/generated/my_campaign/ldm_sft_augmented.jsonl
```

The input is always read-only. Successful responses are appended to
`<output>.checkpoint.jsonl`, keyed by both row index and a content digest. Rerun
the same command after interruption or partial model failure to reuse completed
responses and retry only unfinished records. The key also includes the expert
configuration, so changing the model, endpoint, temperature, or system prompt
does not reuse stale responses. Augmented IR records store the expert model under
`collection.augmentation`; the renderer keeps that provenance out of the prompt.

The augmentation pipeline also accepts Alpaca JSON arrays and JSONL files for
compatibility with older datasets. It updates a JSON target's existing
`reasoning` field when present; otherwise it prepends a `<think>` block without
changing the original answer. Output is always JSONL.

By default the pipeline skips:

- records with an existing non-empty justification
- ldm-2.0 records with `task.reasoning_available: false`
- rendered protein rows, whose source traces do not contain rationale evidence

Use `--overwrite-reasoning` only to intentionally replace existing content.
`--include-reasoning-unavailable` is available for explicit experiments, but it
can fabricate unsupported rationale and conflicts with the default data-quality
rule in `data/SCHEMA.md`.

The CLI reads `LLM_API_KEY` from the environment and never accepts or stores a
credential in source code. It imports `openai` only when the production model
adapter is constructed, so runtime collection remains dependency-light.

## Collection Rule

Collect one row per accepted teacher action:

```text
build model-visible search state
call teacher model
parse response
reject invalid attempts
construct accepted action
append ldm-2.0 IR + rendered SFT row
run BO selection / evaluator
attach outcome metadata if useful
```

Do not collect:

- rejected attempts as training targets
- candidates after BO filtering as if the LLM directly emitted them
- objective predictions made by the model
- prompts that contain leaked answers or stale required outputs
- metadata/provenance inside model-visible instruction text

This matters especially for nanogpt, where response transcripts can contain
multiple rejected tool calls before the accepted operation set.

## Unified Record Shape

All tasks are normalized to:

```json
{
  "schema_version": "ldm-2.0",
  "task": {},
  "search_state": {},
  "request": {},
  "action": {}
}
```

The model input is:

```text
task + search_state + request
```

The model target is:

```text
action
```

`search_state.design_space.representation` determines how to read the action:

- `parameter_edits`: the action edits a parent state, or expands the active
  parameter/feature set.
- `complete_design`: the action emits complete candidate objects.

Action types:

- `propose`: propose candidates in the current design space.
- `expand_design_space`: activate an inactive known dimension.
- `add_new_parameter`: introduce a new primitive not already present in the
  known schema.

## Task Mapping Summary

| Task | IR task id | Domain | Representation | Normal action |
|---|---|---|---|---|
| nanogpt | `nanogpt` | `training_program` | `parameter_edits` | `propose`, `expand_design_space` |
| small molecule | `smallmol` | `molecule` | `complete_design` | `propose` |
| antibody / protein | `protein` | `antibody_sequence` | `complete_design` for direct sequences | `propose` |

## Small Molecule Collection

Small-molecule runtime collection is hooked into the engine campaign export at
[tasks/small_molecule/core/engine_adapters.py](../tasks/small_molecule/core/engine_adapters.py);
the round-record extraction reuses
[tasks/small_molecule/core/ldm_tilted_case2/trace.py](../tasks/small_molecule/core/ldm_tilted_case2/trace.py)'s
IR builder helpers.

Enable it with:

```bash
LDM_DATA_COLLECTION_ENABLED=1 \
python -m tasks.small_molecule.ldm_task.procedure \
  --method m1_stratified_direct_llm_oversample_sir \
  --budget 80 \
  --trajectory-dir runs/case2_real
```

Default output:

```text
tasks/small_molecule/runs/case2_real/ldm_data/ldm_ir.jsonl
tasks/small_molecule/runs/case2_real/ldm_data/ldm_sft.jsonl
tasks/small_molecule/runs/case2_real/ldm_data/dataset_info.json
```

To aggregate many runs into one directory:

```bash
LDM_DATA_COLLECTION_ENABLED=1 \
LDM_DATA_COLLECTION_DIR=/abs/path/to/ldm_collection/smallmol \
python -m tasks.small_molecule.ldm_task.procedure ...
```

What the hook collects:

- accepted `m1_direct` LLM calls from `record["llm_attempts"]`
- the original direct-SMILES user prompt
- the raw accepted JSON response
- parsed `direct_smiles` candidates and rationales
- history summary mapped into role-tagged `observations`
- `avoid_exact_smiles` mapped into `search_state.do_not_repeat`
- diversity alerts mapped into `search_state.progress`

What stays as metadata only:

- selected candidates
- selected scores
- EHVI/SIR probabilities
- drop counts
- trajectory paths

The adapter is intentionally narrow: it handles M1 direct-SMILES proposal calls.
Seed-plan prompts and ReaSyn analog-generation prompts need separate adapters if
they should become SFT data.

## Nanogpt Collection

Nanogpt uses `parameter_edits` because each teacher action either edits an
active `train.py` parameter or expands the active operation-feature space.

Runtime collection is hooked into
`tasks/nanogpt/core/workflow.py`, in `OperationSearchEngine._generate_one`, after:

1. the prompt has been written
2. `_call_operation_generator(...)` returns
3. `validate_generator_action(...)` has accepted the action
4. `operations_payload` has been created from the validated action

The hook does not train from the first tool call in `response.md`; it uses the
accepted `operations_payload`. Enable run-local collection with:

```bash
LDM_DATA_COLLECTION_ENABLED=1 \
python scripts/run_ldm_tts.py config/nanogpt/mock_best_of_n.yaml
```

Unless `LDM_DATA_COLLECTION_DIR` selects a shared campaign directory, the sink
writes `ldm_ir.jsonl`, `ldm_sft.jsonl`, and `dataset_info.json` beneath the
NanoGPT run's `<run_dir>/ldm_data/` directory.

Implemented action mapping:

```python
if action.kind == "feature":
    ir_action = {
        "type": "expand_design_space",
        "reasoning": action.rationale or None,
        "payload": {
            "activate": action.feature.name,
            "initial_value": current_values.get(action.feature.name),
        },
        "summary": f"Activate {action.feature.name}",
    }
else:
    ir_action = {
        "type": "propose",
        "reasoning": " ".join(
            op.rationale for op in (action.operations or []) if op.rationale
        ) or None,
        "payload": {
            "candidates": [{
                "parent": parent.state_id,
                "edits": [
                    {
                        "parameter": op.name,
                        "edit_op": op.op,
                        "value": op.value,
                        "rationale": op.rationale,
                    }
                    for op in (action.operations or [])
                ],
            }]
        },
        "summary": operations_payload.get("summary"),
    }
```

The hook builds and appends:

```python
sink = DataCollectionSink.from_env(default_root=self.config.out_dir / "ldm_data")

ir = make_parameter_edit_ir(
    task_id="nanogpt",
    domain="training_program",
    task_description=self.config.task_context,
    objectives=[{
        "name": self.config.score_key,
        "direction": "minimize" if self.config.minimize else "maximize",
        "description": "Objective measured after executing the candidate train.py.",
    }],
    active_parameters=active_params,
    inactive_parameters=inactive_params,
    action=ir_action,
    request_description="Choose one valid action: propose active edits or expand the design space.",
    max_edits_per_candidate=self.max_operations_per_step,
    round_idx=self.current_iteration,
    observations=observations,
    best_so_far=best_so_far,
    surrogate_feedback=surrogate_feedback,
    progress=progress,
    expansion_history=self.expansion_history,
    applied_this_transition=prior_operations,
    raw_context={"parent_train_py": current_text},
)

sink.append(
    ir,
    provenance={
        "task": "nanogpt",
        "state_id": state.state_id,
        "parent_state_id": parent.state_id,
        "prompt_path": str(state.prompt_path),
        "operations_path": str(operations_path),
    },
)
```

Required field mapping:

- `active_parameters`: derive from `self.operation_schema.parameters`, including
  `current_value` from `extract_top_level_assignment_values(current_text)`.
- `inactive_parameters`: derive from `self.inactive_operation_schema()`.
- `observations`: recent evaluated states and feedback rows, with roles such as
  `recent`, `best`, `best_path`, or `evaluated`.
- `best_so_far`: current best real evaluated state.
- `surrogate_feedback`: GP prediction/acquisition information that was visible
  to the teacher for this proposal.
- `progress`: rounds since improvement or other stall signals.
- `raw_context.parent_train_py`: include the parent artifact when the inference
  environment will also show it.

For future L2 feature invention, map unknown proposed features to:

```json
{
  "type": "add_new_parameter",
  "payload": {
    "parameter": {"name": "...", "type": "...", "domain": "..."},
    "code_sketch": null,
    "why_new_axis": "..."
  }
}
```

Only use `add_new_parameter` when the proposed feature was not already in the
known inactive schema.

## Antibody / Protein Collection

The direct CDRH3 sequence task uses `complete_design`:

```text
history + antigen context + constraints + request  ->  sequence candidate(s)
```

The basic IR mapping is:

- `task.id`: `protein`
- `task.domain`: `antibody_sequence`
- `objectives`: `binding_energy` or `absolut_energy`, direction `minimize`
- `design_space.representation`: `complete_design`
- `active_parameters`: one `sequence` parameter with fixed length and alphabet
- `allows_new_parameters`: `false`
- `do_not_repeat`: observed sequences
- `action.type`: `propose`
- `action.payload.candidates`: emitted sequences
- `reasoning_available`: `false` for sequence-only traces with no rationale

Runtime collection is hooked into
`tasks/antibody/core/ldm_light/ldm_acq.py` in `run_one`, after a direct proposal
path returns and before `evaluator.energy(...)` is called. For acquisition-guided
direct generation, the target contains the full validated LLM candidate set from
before acquisition selection, not only the candidates selected using hidden GP
scores. This covers direct initialization, pure LLM generation, direct
generation with downstream acquisition, and the legacy model-selected candidate
pool. It deliberately skips random fallback and post-warmup policy/DSL actions.

Enable it with any antibody config, for example:

```bash
LDM_DATA_COLLECTION_ENABLED=1 \
python scripts/run_ldm_tts.py config/antibody/mock_ei.yaml
```

The default output is `<run_dir>/ldm_data/{ldm_ir.jsonl,ldm_sft.jsonl,dataset_info.json}`.
Set `LDM_DATA_COLLECTION_DIR` to aggregate multiple antigen/seed runs into one
campaign; provenance retains the antigen, seed, method, source, and run path.

The hook builds direct-sequence IR like:

```python
ir = make_complete_design_ir(
    task_id="protein",
    domain="antibody_sequence",
    task_description=(
        "Direct CDRH3 antibody sequence generation for AntBO. "
        "Generate antibody strings directly."
    ),
    objectives=[{
        "name": "binding_energy",
        "direction": "minimize",
        "description": "Absolut binding energy; lower is better.",
    }],
    design_space_description="Fixed-length CDRH3 sequence with developability constraints.",
    active_parameters=[{
        "name": "sequence",
        "type": "string",
        "domain": {"length": seq_len, "alphabet": list(AA)},
        "edit_op": None,
    }],
    observations=[
        {
            "design": row["LastProtein"],
            "results": {"binding_energy": row["LastValue"]},
            "roles": ["recent"],
            "round": row["Index"],
        }
        for row in rows[-args.history_top_k:]
    ],
    best_so_far=best_so_far,
    candidates=[
        {"design": candidate["sequence"], "rationale": None}
        for candidate in selected_candidates
    ],
    request_description=(
        f"Propose {batch_size} CDRH3 sequence(s) of length {seq_len} "
        f"over alphabet {AA}; do not repeat observed sequences."
    ),
    num_candidates=batch_size,
    num_evaluated=len(rows),
    do_not_repeat=sorted(observed)[-200:],
    allows_new_parameters=False,
    reasoning_available=False,
    raw_context={
        "target_id": antigen,
        "target_context": antigen_context or {},
        "candidate_pool": candidate_pool,
    },
)
```

Then append:

```python
sink.append(
    ir,
    provenance={
        "task": "protein",
        "antigen": antigen,
        "seed": seed,
        "eval_start": eval_idx,
        "source": decision.get("source"),
    },
)
```

Important antibody-specific rules:

- Drop stale or answer-leaking `required_output` fields.
- Do not fabricate rationales for sequence-only traces.
- Do not collect fallback-random candidates as teacher demonstrations unless
  they are explicitly labeled and intentionally included for robustness.
- If the model is asked to choose from a `candidate_pool`, include the pool in
  model-visible state because the action is not valid without it.

### Post-Warmup Antibody DSL Mode

After warmup, `select_with_parallel_ldm(...)` calls the LDM orchestrator to
produce DSL updates such as trust-region search atoms. That is not the same
semantic action as direct sequence proposal.

Recommended handling:

1. Collect direct sequence warmup examples as `protein` / `complete_design`.
2. For DSL updates, either:
   - introduce a separate action type such as `update_search_policy`, or
   - map only clearly space-expanding trust-region updates into
     `expand_design_space` with an explicit DSL payload.
3. Do not mix DSL update targets with direct sequence targets under the same
   natural-language request unless the model will see exactly that mixed
   contract at inference.

If collecting DSL mode now, keep it in a separate dataset shard:

```text
ldm_ir_antibody_sequence.jsonl
ldm_ir_antibody_dsl.jsonl
```

Merge only after the action contract is stable.

## Historical Trace Conversion

The scripts in `data/` convert historical samples and nanogpt run
directories into `ldm-2.0` IR.

Sample bundle:

```bash
python data/build_ldm2.py from-sample \
  --in /path/to/ldm_data_sample.json \
  --out-ir data/generated/imported/ldm_ir.jsonl
```

Full nanogpt run:

```bash
python data/build_ldm2.py from-nanogpt-run \
  --run-dir /path/to/expanded_ldm_bon_N4H4_03 \
  --out-ir data/generated/imported/ldm_ir.jsonl \
  --min-status evaluated
```

Render IR into Alpaca:

```bash
python data/build_ldm2.py render \
  --in-ir data/generated/imported/ldm_ir.jsonl \
  --out data/generated/imported/ldm_sft.jsonl \
  --render prose \
  --dataset-info data/generated/imported/dataset_info.json
```

Audit and verify before training:

```bash
python data/build_ldm2.py audit \
  --in-ir data/generated/imported/ldm_ir.jsonl

python data/verify.py all \
  --run-dir /path/to/run \
  --in-ir data/generated/imported/ldm_ir.jsonl \
  --sft data/generated/imported/ldm_sft.jsonl \
  --dataset-info data/generated/imported/dataset_info.json \
  --cutoff-len 16384
```

## Quality Gates

Run at least:

```bash
python -m pytest \
  tests/test_data_collection.py \
  tests/test_data_augmentation.py \
  tests/test_ldm_tts_core.py
```

For generated datasets, check:

- every `action.type` is allowed by `request.allowed_actions`
- all proposed parameters are active, or activated parameters are inactive
- no `collection`, `provenance`, `required_output`, or legacy tool-contract text
  leaks into rendered instructions
- `do_not_repeat` never conflicts with the target action
- nanogpt records use accepted operations, not rejected transcript attempts
- protein records keep missing rationales as `null`
- context length fits the intended `cutoff_len`

## Dataset Splitting

Recommended split discipline:

- Split by run/antigen/seed, not by individual row, when possible.
- Keep synthetic augmentation out of validation and test sets.
- Keep antibody direct-sequence and antibody DSL records separate until their
  action contracts are finalized.
- For nanogpt, report action distribution and edited-parameter concentration.
  If `expand_design_space` is below roughly 5 percent, a plain SFT model will
  probably ignore that behavior.

## Fine-Tuning Use

For LlamaFactory, place:

```text
ldm_sft.jsonl
dataset_info.json
```

in its `data/` directory or merge the `dataset_info.json` entry into the
existing registry.

For the repository's full-parameter Qwen rationale-distillation recipe, keep the
augmented IR in `data/generated/<campaign>/` and use
[`finetune/prepare_dataset.py`](../finetune/prepare_dataset.py) to create
provenance-grouped train/evaluation shards plus their registry under
`data/generated/full_sft/`. See [`finetune/README.md`](../finetune/README.md) for
the complete validation, training, and inference-parity workflow.

Use a cutoff length that fits the rendered prompt. For nanogpt with embedded
`train.py`, use a large cutoff such as `16384`. If you use
`LDM_DATA_COLLECTION_STRIP_PARENT_ARTIFACT=1`, document the train/inference
mismatch and evaluate it separately.

## Inference-Time Parity

The fine-tuned model should be prompted with the same renderer used for SFT:

```python
from ldm_tts.data import render_prose
```

Do not train on `ldm-2.0` rendered prompts and then deploy against the legacy
task-specific prompt format. That mismatch is large enough to make the
fine-tuned proposer fail even when the dataset itself is valid.

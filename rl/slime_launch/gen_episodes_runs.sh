#!/bin/bash
# Generate the episode prompt-data for every run in the matrix (see RUNS.md).
# One file PER RUN (R1-R4), not per reward: R1 and R2 share a reward but are
# different runs, so each must get its OWN gp_history_file + output_dir. Sharing
# a GP or output dir across runs couples their rewards and breaks reproducibility
# (see KangOxford PR #2 / notebook 06). The model is chosen at launch time, not here.
set -eu
REPO_ROOT=/mnt/data0/ys/LDM
CONFIG=$REPO_ROOT/rl/slime_launch/config_real.json
export PYTHONPATH=$REPO_ROOT/rl:$REPO_ROOT:${PYTHONPATH:-}
EP=rl.ldm_rl.episodes

read -r COUNT ITERS RES EVALS WARMUP WARMUP_ITERS < <(python3 -c "
import json;c=json.load(open('$CONFIG'));e=c['episodes'];w=c['warmup']
print(e['count'],e['iterations'],e['reservoir_size'],e['evaluations_per_round'],w['num_samples'],w['iterations'])")
BASE_GP=$(python3 -c "import json;print(json.load(open('$CONFIG'))['gp_history_file'])")

GP_DIR=$REPO_ROOT/rl/gp_history      # per-run GP files live here
OUT_DIR=$REPO_ROOT/rl/run_out        # per-run docking output dirs live here
mkdir -p "$GP_DIR" "$OUT_DIR"

# real_kwargs = config.real_kwargs but with a per-run gp_history_file + output_dir
rk_json() {  # rk_json <gp_history_file> <output_dir>
  python3 -c "
import json,sys;c=json.load(open('$CONFIG'))
rk=dict(c['real_kwargs']);rk['gp_history_file']=sys.argv[1];rk['output_dir']=sys.argv[2]
print(json.dumps(rk))" "$1" "$2"
}

gen() {  # gen <out.jsonl> <reward> <agg> <count> <iters> <gp_history_file> <output_dir>
  python3 -m $EP --output "$REPO_ROOT/$1" --task small_molecule --mode real \
    --count "$4" --iterations "$5" --reservoir-size "$RES" --evaluations-per-round "$EVALS" \
    --reward "$2" --acquisition-agg "$3" --real-kwargs "$(rk_json "$6" "$7")"
}

# warm-up (rollout-only) writes the shared BASE GP; each run is then seeded from it.
gen rl_episodes_sm_warmup.jsonl acquisition max "$WARMUP" "$WARMUP_ITERS" "$BASE_GP" "$OUT_DIR/warmup"

# One episode file PER RUN, each with its own GP + output dir.
# R1 base / R2 SFT share the acquisition-max reward but stay isolated (own GP).
gen rl_episodes_sm_R1.jsonl acquisition max  "$COUNT" "$ITERS" "$GP_DIR/R1.jsonl" "$OUT_DIR/R1"
gen rl_episodes_sm_R2.jsonl acquisition max  "$COUNT" "$ITERS" "$GP_DIR/R2.jsonl" "$OUT_DIR/R2"
gen rl_episodes_sm_R3.jsonl hypervolume max  "$COUNT" "$ITERS" "$GP_DIR/R3.jsonl" "$OUT_DIR/R3"
gen rl_episodes_sm_R4.jsonl acquisition mean "$COUNT" "$ITERS" "$GP_DIR/R4.jsonl" "$OUT_DIR/R4"

echo "episodes -> $REPO_ROOT/rl_episodes_sm_{warmup,R1,R2,R3,R4}.jsonl"
echo "per-run GP under $GP_DIR/ , outputs under $OUT_DIR/"
echo "BEFORE launching each run, seed its GP from the warm base:"
echo "  for r in R1 R2 R3 R4; do cp \"$BASE_GP\" \"$GP_DIR/\$r.jsonl\"; done"

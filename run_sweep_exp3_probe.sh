#!/bin/bash

#SBATCH --job-name=ti-probe3
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-239
#SBATCH --output=logs/probe3_%A_%a.out
#SBATCH --error=logs/probe3_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# ── Model table ────────────────────────────────────────────────────────────────
MODELS=(
    "Qwen/Qwen3.5-2B-Base"
    "Qwen/Qwen3.5-4B-Base"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
)
MODEL_SLUGS=("qwen35-2b" "qwen35-4b" "llama32-1b" "llama32-3b")
N_MODELS=4

# ── Flat (task, hierarchy) pair table ─────────────────────────────────────────
# EW=-1 / EL=-1  →  no exception (pure TI).
TH_TASK_NAME=( sports  sports      sports       sports       sports  sports      sports       sports       poker_equity  poker_equity )
TH_TEAM_TYPE=( fake    fake        fake         fake         real    real        real         real         ""            "" )
TH_TASK_SLUG=( fake    fake        fake         fake         real    real        real         real         poker         poker )
TH_HIER_TAG=(  ti_n9   exc_mid_n9  exc_mid_n11  exc_far_n11  ti_n9   exc_mid_n9  exc_mid_n11  exc_far_n11  ti_n9         exc_mid_n9 )
TH_N=(          9       9           11           11           9       9           11           11           9             9 )
TH_EW=(        -1       5            6            7          -1       5            6            7          -1             5 )
TH_EL=(        -1       3            4            3          -1       3            4            3          -1             3 )
N_TH=10

# ── L2 regularisation values ───────────────────────────────────────────────────
L2_REGS=("1e-4" "0.1" "1.0" "10.0" "20.0" "50.0")
N_L2=6

# ── Decode array index ─────────────────────────────────────────────────────────
TASK_ID=$SLURM_ARRAY_TASK_ID

l2_idx=$(( TASK_ID % N_L2 ))
rem=$(( TASK_ID / N_L2 ))
th_idx=$(( rem % N_TH ))
model_idx=$(( rem / N_TH ))

MODEL="${MODELS[$model_idx]}"
MODEL_SLUG="${MODEL_SLUGS[$model_idx]}"
TASK_NAME="${TH_TASK_NAME[$th_idx]}"
TEAM_TYPE="${TH_TEAM_TYPE[$th_idx]}"
TASK_SLUG="${TH_TASK_SLUG[$th_idx]}"
HIER_TAG="${TH_HIER_TAG[$th_idx]}"
N="${TH_N[$th_idx]}"
EW="${TH_EW[$th_idx]}"
EL="${TH_EL[$th_idx]}"
L2="${L2_REGS[$l2_idx]}"
# Filesystem-safe slug: replace '.' with 'p' and 'e' notation stays as-is
L2_SLUG="${L2//./p}"   # e.g. 1e-4 → 1e-4, 0.1 → 0p1, 10.0 → 10p0

echo "=== Job ${SLURM_ARRAY_JOB_ID}_${TASK_ID} ==="
echo "  model=${MODEL}  task=${TASK_SLUG}  hier=${HIER_TAG}  l2=${L2}"

# ── Inner loop: 100 configs (seed 0–49 × fwd/rev) ─────────────────────────────
for INNER_ID in $(seq 0 99); do

    SEED=$(( INNER_ID % 50 ))
    REVERSE_IDX=$(( INNER_ID / 50 ))

    if [ "$REVERSE_IDX" -eq 1 ]; then
        REVERSE_FLAG="--reverse"
        REV_TAG="rev"
    else
        REVERSE_FLAG=""
        REV_TAG="fwd"
    fi

    OUTPUT_DIR="results/sweep3_probe/${MODEL_SLUG}/${TASK_SLUG}/${HIER_TAG}/l2_${L2_SLUG}/seed${SEED}_${REV_TAG}"

    if [ -f "${OUTPUT_DIR}/results.json" ]; then
        echo "  [skip] seed=${SEED} ${REV_TAG}"
        continue
    fi

    mkdir -p "$OUTPUT_DIR"
    echo "  [run ] seed=${SEED} ${REV_TAG}  →  ${OUTPUT_DIR}"

    CMD=(
        python sports_hierarchy_finetuning.py
            --finetune-method probing
            --probe-epochs    50
            --probe-l2-reg    "$L2"
            --shuffle-train
            --task            "$TASK_NAME"
            --model           "$MODEL"
            --n               "$N"
            --seed            "$SEED"
            --output-dir      "$OUTPUT_DIR"
    )

    # --team-type only applies to sports tasks
    if [ -n "$TEAM_TYPE" ]; then
        CMD+=(--team-type "$TEAM_TYPE")
    fi

    # --exception only for non-TI configs
    if [ "$EW" -ge 0 ]; then
        CMD+=(--exception "$EW" "$EL")
    fi

    if [ -n "$REVERSE_FLAG" ]; then
        CMD+=("$REVERSE_FLAG")
    fi

    "${CMD[@]}"

done

echo "=== Array element ${TASK_ID} complete ==="

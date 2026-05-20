#!/bin/bash

#SBATCH --job-name=ti-full
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-23
#SBATCH --output=logs/full_%A_%a.out
#SBATCH --error=logs/full_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# ── Model table ────────────────────────────────────────────────────────────────
MODELS=("Qwen/Qwen3.5-2B-Base" "meta-llama/Llama-3.2-1B")
MODEL_SLUGS=("qwen35-2b" "llama32-1b")
MODEL_LRS=("1e-6" "2e-6")
N_MODELS=2

# ── Flat (task, hierarchy) pair table ─────────────────────────────────────────
# EW=-1 / EL=-1  →  no exception (pure TI).
TH_TASK_NAME=( sports      sports      sports       sports       sports  sports      sports       sports       poker_equity  poker_equity  poker_equity  poker_equity )
TH_TEAM_TYPE=( fake        fake        fake         fake         real    real        real         real         ""            ""            ""            "" )
TH_TASK_SLUG=( fake        fake        fake         fake         real    real        real         real         poker         poker         poker         poker )
TH_HIER_TAG=(  ti_n9       exc_mid_n9  exc_mid_n11  exc_far_n11  ti_n9   exc_mid_n9  exc_mid_n11  exc_far_n11  ti_n9         exc_mid_n9    exc_mid_n11   exc_far_n11 )
TH_N=(          9           9           11           11           9       9           11           11           9             9             11            11 )
TH_EW=(        -1           5            6            7          -1       5            6            7          -1             5              6             7 )
TH_EL=(        -1           3            4            3          -1       3            4            3          -1             3              4             3 )
N_TH=12

# ── Decode array index ─────────────────────────────────────────────────────────
TASK_ID=$SLURM_ARRAY_TASK_ID

th_idx=$(( TASK_ID % N_TH ))
model_idx=$(( TASK_ID / N_TH ))

MODEL="${MODELS[$model_idx]}"
MODEL_SLUG="${MODEL_SLUGS[$model_idx]}"
LR="${MODEL_LRS[$model_idx]}"
TASK_NAME="${TH_TASK_NAME[$th_idx]}"
TEAM_TYPE="${TH_TEAM_TYPE[$th_idx]}"
TASK_SLUG="${TH_TASK_SLUG[$th_idx]}"
HIER_TAG="${TH_HIER_TAG[$th_idx]}"
N="${TH_N[$th_idx]}"
EW="${TH_EW[$th_idx]}"
EL="${TH_EL[$th_idx]}"

echo "=== Job ${SLURM_ARRAY_JOB_ID}_${TASK_ID} ==="
echo "  model=${MODEL}  lr=${LR}  task=${TASK_SLUG}  hier=${HIER_TAG}"

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

    OUTPUT_DIR="results/sweep_full/${MODEL_SLUG}/${TASK_SLUG}/${HIER_TAG}/seed${SEED}_${REV_TAG}"

    if [ -f "${OUTPUT_DIR}/results.json" ]; then
        echo "  [skip] seed=${SEED} ${REV_TAG}"
        continue
    fi

    mkdir -p "$OUTPUT_DIR"
    echo "  [run ] seed=${SEED} ${REV_TAG}  →  ${OUTPUT_DIR}"

    CMD=(
        python sports_hierarchy_finetuning.py
            --finetune-method full
            --n-epochs        100
            --batch-size      32
            --shuffle-train
            --lr              "$LR"
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

#!/bin/bash

#SBATCH --job-name=ti-sweep3
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-199
#SBATCH --output=logs/sweep3_%A_%a.out
#SBATCH --error=logs/sweep3_%A_%a.err

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
# Parallel arrays — each index defines one valid (task, hier) combination.
# EW=-1 / EL=-1  →  no exception (pure TI).
TH_TASK_NAME=( sports  sports      sports       sports       sports  sports      sports       sports       poker_equity  poker_equity )
TH_TEAM_TYPE=( fake    fake        fake         fake         real    real        real         real         ""            "" )
TH_TASK_SLUG=( fake    fake        fake         fake         real    real        real         real         poker         poker )
TH_HIER_TAG=(  ti_n9   exc_mid_n9  exc_mid_n11  exc_far_n11  ti_n9   exc_mid_n9  exc_mid_n11  exc_far_n11  ti_n9         exc_mid_n9 )
TH_N=(          9       9           11           11           9       9           11           11           9             9 )
TH_EW=(        -1       5            6            7          -1       5            6            7          -1             5 )
TH_EL=(        -1       3            4            3          -1       3            4            3          -1             3 )
N_TH=10

# ── LoRA ranks ─────────────────────────────────────────────────────────────────
LORA_RS=(4 8 16 32 64)
N_R=5

# ── Decode array index ─────────────────────────────────────────────────────────
TASK_ID=$SLURM_ARRAY_TASK_ID

r_idx=$(( TASK_ID % N_R ))
rem=$(( TASK_ID / N_R ))
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
LORA_R="${LORA_RS[$r_idx]}"
LORA_ALPHA=$(( 2 * LORA_R ))

echo "=== Job ${SLURM_ARRAY_JOB_ID}_${TASK_ID} ==="
echo "  model=${MODEL}  task=${TASK_SLUG}  hier=${HIER_TAG}  lora_r=${LORA_R}  lora_alpha=${LORA_ALPHA}"

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

    OUTPUT_DIR="results/sweep3/${MODEL_SLUG}/${TASK_SLUG}/${HIER_TAG}/lora_r${LORA_R}_alpha${LORA_ALPHA}/seed${SEED}_${REV_TAG}"

    if [ -f "${OUTPUT_DIR}/results.json" ]; then
        echo "  [skip] seed=${SEED} ${REV_TAG}"
        continue
    fi

    mkdir -p "$OUTPUT_DIR"
    echo "  [run ] seed=${SEED} ${REV_TAG}  →  ${OUTPUT_DIR}"

    CMD=(
        python sports_hierarchy_finetuning.py
            --finetune-method lora
            --n-epochs        10
            --shuffle-train
            --task            "$TASK_NAME"
            --model           "$MODEL"
            --n               "$N"
            --lora-r          "$LORA_R"
            --lora-alpha      "$LORA_ALPHA"
            --lora-modules    q_proj v_proj k_proj o_proj up_proj down_proj
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

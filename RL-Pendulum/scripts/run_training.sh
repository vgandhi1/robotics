#!/usr/bin/env bash
# run_training.sh — Full RL-Pendulum training pipeline
#
# Phase 2: Baseline PPO (no DR), 5M steps
# Phase 3: PPO with Domain Randomization, 10M steps
#
# Usage:
#   bash scripts/run_training.sh                    # Full pipeline
#   bash scripts/run_training.sh --phase2-only      # Only baseline
#   bash scripts/run_training.sh --phase3-only      # Only DR training
#   bash scripts/run_training.sh --resume <path>    # Resume from checkpoint

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG="configs/ppo_config.yaml"
SEED=42
PHASE2_ONLY=false
PHASE3_ONLY=false
RESUME_PATH=""

# ── Argument parsing ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase2-only) PHASE2_ONLY=true; shift ;;
        --phase3-only) PHASE3_ONLY=true; shift ;;
        --resume)      RESUME_PATH="$2"; shift 2 ;;
        --seed)        SEED="$2"; shift 2 ;;
        --config)      CONFIG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Activate virtualenv if present ─────────────────────────────────────────
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "============================================================"
echo "  RL-Pendulum Training Pipeline"
echo "  Config: $CONFIG"
echo "  Seed:   $SEED"
echo "============================================================"

# ── Phase 2: Baseline (no DR) ───────────────────────────────────────────────
if [ "$PHASE3_ONLY" = false ]; then
    echo ""
    echo "── Phase 2: Baseline PPO (no domain randomization) ──────────────"
    python training/train.py \
        --config "$CONFIG" \
        --no-dr \
        --seed "$SEED"
    echo "Phase 2 complete. Best model: logs/best_model.zip"
fi

# ── Phase 3: DR training ─────────────────────────────────────────────────────
if [ "$PHASE2_ONLY" = false ]; then
    echo ""
    echo "── Phase 3: PPO with Domain Randomization ───────────────────────"

    if [ -n "$RESUME_PATH" ]; then
        RESUME_ARG="--resume $RESUME_PATH"
    elif [ -f "logs/best_model.zip" ]; then
        RESUME_ARG="--resume logs/best_model.zip"
        echo "  Resuming from Phase 2 checkpoint: logs/best_model.zip"
    else
        RESUME_ARG=""
    fi

    python training/train.py \
        --config "$CONFIG" \
        --seed "$SEED" \
        $RESUME_ARG
    echo "Phase 3 complete. Best DR model: logs/best_model.zip"
fi

echo ""
echo "============================================================"
echo "  Training complete!"
echo "  Models saved to: logs/"
echo "  TensorBoard:  tensorboard --logdir logs/tensorboard"
echo "  Next step:    bash scripts/export_model.sh"
echo "============================================================"

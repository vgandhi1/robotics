#!/usr/bin/env bash
# export_model.sh — Export, quantize, and prepare RL policy for ESP32
#
# Steps:
#   1. PyTorch → ONNX FP32   (via torch.onnx.export)
#   2. ONNX FP32 → INT8 ONNX (via onnxruntime dynamic quantization)
#   3. ONNX → TFLite INT8    (via tf2onnx + TFLite converter) [optional]
#   4. TFLite → C header     (via xxd, embedded in firmware)
#
# Usage:
#   bash scripts/export_model.sh                         # uses logs/best_model.zip
#   bash scripts/export_model.sh --model logs/final_model.zip
#   bash scripts/export_model.sh --no-tflite             # skip TF step

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MODEL_PATH="logs/best_model.zip"
ONNX_PATH="export/model.onnx"
INT8_PATH="export/model_int8.onnx"
TFLITE_PATH="export/model.tflite"
HEADER_PATH="firmware/inference_loop/rl_policy_data.h"
SKIP_TFLITE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL_PATH="$2"; shift 2 ;;
        --no-tflite)  SKIP_TFLITE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

mkdir -p export

echo "============================================================"
echo "  RL-Pendulum Model Export Pipeline"
echo "  Source model : $MODEL_PATH"
echo "============================================================"

# ── Step 1: ONNX FP32 ───────────────────────────────────────────────────────
echo ""
echo "── Step 1: Export to ONNX FP32 ─────────────────────────────────────"
python export/export_onnx.py \
    --model "$MODEL_PATH" \
    --output "$ONNX_PATH"

# ── Step 2: INT8 ONNX ───────────────────────────────────────────────────────
echo ""
echo "── Step 2: Quantize to INT8 ─────────────────────────────────────────"
python export/quantize.py \
    --onnx "$ONNX_PATH" \
    --output-int8 "$INT8_PATH" \
    $([ "$SKIP_TFLITE" = true ] && echo "" || echo "--output-tflite $TFLITE_PATH")

# ── Step 3: Generate C header ────────────────────────────────────────────────
if [ "$SKIP_TFLITE" = false ] && [ -f "$TFLITE_PATH" ]; then
    echo ""
    echo "── Step 3: Generate C header for ESP32 ─────────────────────────────"
    python -c "
from export.quantize import generate_c_header
generate_c_header('$TFLITE_PATH', '$HEADER_PATH')
"
    echo "  Header: $HEADER_PATH"
else
    echo ""
    echo "── Step 3: Generating C header from ONNX INT8 (no TFLite) ──────────"
    # Fallback: encode INT8 ONNX directly as a C array
    if command -v xxd &>/dev/null; then
        echo "// Auto-generated from $INT8_PATH" > "$HEADER_PATH"
        echo "// DO NOT EDIT MANUALLY" >> "$HEADER_PATH"
        echo "" >> "$HEADER_PATH"
        xxd -i "$INT8_PATH" >> "$HEADER_PATH"
        echo "  Header (ONNX INT8): $HEADER_PATH"
    else
        echo "  [WARN] xxd not found; header not generated. Run manually:"
        echo "         xxd -i $INT8_PATH > $HEADER_PATH"
    fi
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Export complete!"
echo ""
echo "  Files generated:"
[ -f "$ONNX_PATH"   ] && echo "  ✓ $ONNX_PATH   ($(du -k "$ONNX_PATH"   | cut -f1) KB)"
[ -f "$INT8_PATH"   ] && echo "  ✓ $INT8_PATH   ($(du -k "$INT8_PATH"   | cut -f1) KB)"
[ -f "$TFLITE_PATH" ] && echo "  ✓ $TFLITE_PATH ($(du -k "$TFLITE_PATH" | cut -f1) KB)"
[ -f "$HEADER_PATH" ] && echo "  ✓ $HEADER_PATH ($(du -k "$HEADER_PATH" | cut -f1) KB)"
echo ""
echo "  Next step: Flash firmware"
echo "    Open firmware/inference_loop/inference_loop.ino in Arduino IDE"
echo "============================================================"

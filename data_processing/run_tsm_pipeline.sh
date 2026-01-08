#!/usr/bin/env bash
set -euo pipefail

PYTHON=python3
SCRIPT="tsm.py"

LOG_DIR="../processed/logs"
mkdir -p "$LOG_DIR"

TRAIN_LOG="$LOG_DIR/train.log"
TEST_LOG="$LOG_DIR/test.log"

echo "=============================="
echo " LIAD PIPELINE START"
echo "=============================="

echo
echo ">>> Running TRAIN phase..."
echo "Logging to: $TRAIN_LOG"
$PYTHON "$SCRIPT" 1 | tee "$TRAIN_LOG"

echo
echo ">>> TRAIN completed successfully."
echo

echo ">>> Running TEST phase..."
echo "Logging to: $TEST_LOG"
$PYTHON "$SCRIPT" 2 | tee "$TEST_LOG"

echo
echo "=============================="
echo "  ALL DONE SUCCESSFULLY"
echo "=============================="



#to run ./run_tsm_pipeline.sh 
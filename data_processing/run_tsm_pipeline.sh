#!/usr/bin/env bash
set -euo pipefail
PYTHON=python3
SCRIPT="tsm.py"

LOG_DIR="../processed/logs"
mkdir -p "$LOG_DIR"

TRAIN_LOG="$LOG_DIR/train.log"
TEST_LOG="$LOG_DIR/test.log"

echo "====================================="
echo " LIAD Data processing PIPELINE START"
echo "====================================="
echo "Warning: can run upto multiple hours depending on the amount of agents in the data, please refer to read.me"

echo
echo ">>> Running TRAIN phase..."
echo "Logging to: $TRAIN_LOG"
$PYTHON "$SCRIPT" 1 | tee "$TRAIN_LOG"

echo
echo ">>> TRAIN completed successfully."
echo

#echo ">>> Running TEST phase..."
#echo "Logging to: $TEST_LOG"
#$PYTHON "$SCRIPT" 2 | tee "$TEST_LOG"

echo
echo "=============================="
echo "  ALL DONE SUCCESSFULLY"
echo "=============================="

# Remarks: Running this script completely will generate train monthly and test monthly summaries, in order
    # to check the running time it is recommended to comment out the Test phase first.

# 1. Make it an executable script 
    # chmod +x run_tsm_pipeline.sh

# 2. Run the script ./run_tsm_pipeline.sh
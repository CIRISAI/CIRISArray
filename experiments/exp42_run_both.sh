#!/bin/bash
#
# Experiment 42: Cross-Device Correlation Test
#
# This script coordinates capture on both devices and analyzes correlation.
#
# Usage:
#   ./exp42_run_both.sh [duration_seconds]
#
# Prerequisites:
#   - SSH key access to emoore@jetson.local
#   - exp42_jetson_receiver.py copied to Jetson
#

DURATION=${1:-60}
JETSON_HOST="emoore@jetson.local"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "CROSS-DEVICE CIRISARRAY EXPERIMENT"
echo "========================================"
echo ""
echo "Duration: ${DURATION}s"
echo "Jetson: ${JETSON_HOST}"
echo ""

# First, copy the Jetson script to the Jetson
echo "Copying Jetson receiver script..."
scp "${SCRIPT_DIR}/exp42_jetson_receiver.py" "${JETSON_HOST}:/tmp/"

echo ""
echo "Starting captures in 5 seconds..."
echo "Both devices will capture simultaneously."
echo ""
sleep 2

# Start Jetson capture in background via SSH
echo "Starting Jetson capture..."
ssh "${JETSON_HOST}" "python3 /tmp/exp42_jetson_receiver.py --duration ${DURATION} --output /tmp/jetson_capture.npz" &
JETSON_PID=$!

# Small delay to let Jetson start
sleep 1

# Start 4090 capture
echo "Starting 4090 capture..."
python3 "${SCRIPT_DIR}/exp42_4090_receiver.py" --duration ${DURATION} --output /tmp/4090_capture.npz

# Wait for Jetson to finish
echo ""
echo "Waiting for Jetson to complete..."
wait $JETSON_PID

# Copy Jetson results back
echo ""
echo "Copying Jetson results..."
scp "${JETSON_HOST}:/tmp/jetson_capture.npz" /tmp/

# Analyze correlation
echo ""
echo "Analyzing cross-device correlation..."
python3 "${SCRIPT_DIR}/exp42_4090_receiver.py" --analyze /tmp/4090_capture.npz /tmp/jetson_capture.npz

echo ""
echo "Done! Results saved in /tmp/"

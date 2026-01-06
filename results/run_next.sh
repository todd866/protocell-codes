#!/bin/bash
# Wait for current simulation to finish, then run a bigger one

LOG_DIR="/Users/iantodd/Projects/highdimensional/biology/60_heroX_evolution/simulation"
cd "$LOG_DIR"

echo "[$(date '+%H:%M:%S')] Waiting for massive simulation to finish..."

# Wait for process to finish
while pgrep -f "overnight_run.py" > /dev/null; do
    sleep 60
done

echo "[$(date '+%H:%M:%S')] Massive simulation complete. Starting extreme scale..."

# Run extreme scale (217 vesicles = 8 rings, 768D)
python3 -c "
import sys
sys.path.insert(0, '.')

# Patch the scale params before importing
import overnight_run as runner

# Override to extreme scale
runner.N_INTERNAL = 768
runner.N_READOUT = 120  
runner.N_TRIALS = 25

# Rebuild topology for 8 rings
runner.N_RINGS = 8
runner.N_VESICLES, runner.NEIGHBORS = runner.build_hex_neighbors(8)
runner.CENTER_IDX, runner.EDGE_IDX = runner.get_spatial_regions(runner.N_VESICLES, runner.NEIGHBORS)

print(f'EXTREME SCALE: {runner.N_VESICLES} vesicles, {runner.N_INTERNAL}D, {runner.N_TRIALS} trials')

# Run
runner.main()
" > overnight_extreme_$(date +%Y%m%d_%H%M%S).log 2>&1

echo "[$(date '+%H:%M:%S')] Extreme simulation complete."

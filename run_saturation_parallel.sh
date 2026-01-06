#!/bin/bash
# Run saturation experiment in parallel across environment counts
# Use on a multi-core machine (Codespace, cloud, etc.)

pip install -r requirements.txt

# Create output directory
mkdir -p results/saturation_parallel

# Run each environment count in parallel
for n_envs in 8 16 32 48 64 96 128 192 256; do
    echo "Starting N=$n_envs..."
    python -c "
import sys
sys.path.insert(0, '.')
from capacity_saturation import *
import json

n_envs = $n_envs
print(f'Running N={n_envs}')

configs = generate_environment_configs(n_envs)
env_to_codes = collect_codes_for_environments(configs, n_repeats=3)
results = count_distinguishable_attractors(env_to_codes, method='all')
results['n_environments'] = n_envs

with open(f'results/saturation_parallel/n{n_envs}.json', 'w') as f:
    json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

print(f'Done N={n_envs}: silhouette_k={results.get(\"silhouette_k\")}, accuracy={results.get(\"decoding_accuracy\", 0)*100:.1f}%')
" &
done

# Wait for all to finish
wait
echo "All done! Results in results/saturation_parallel/"

# Combine results
python -c "
import json
from pathlib import Path
results = []
for f in sorted(Path('results/saturation_parallel').glob('n*.json')):
    with open(f) as fp:
        results.append(json.load(fp))
results.sort(key=lambda x: x['n_environments'])
print('\\n=== SATURATION SUMMARY ===')
print('N_envs\\tSilhouette_k\\tDecoding_acc')
for r in results:
    print(f\"{r['n_environments']}\\t{r.get('silhouette_k', 'N/A')}\\t{r.get('decoding_accuracy', 0)*100:.1f}%\")
with open('results/saturation_parallel/combined.json', 'w') as f:
    json.dump(results, f, indent=2)
"

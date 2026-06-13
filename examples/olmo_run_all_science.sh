#!/usr/bin/env bash
# Run all OLMo frontier science scripts sequentially on MSI.
# Called from coordinator: /Users/user/msi-node/msi 'bash /path/to/this/script'

set -euo pipefail

VENV=/projects/standard/hsiehph/sauer354/gamfit-sweep-venv
GAM=/projects/standard/hsiehph/sauer354/gam
OUTDIR=/projects/standard/hsiehph/sauer354/olmo_data/plots

source "$VENV/bin/activate"
cd /tmp  # avoid import collisions

echo "=== CURVATURE SCIENCE ==="
python "$GAM/examples/olmo_curvature_science.py" 2>&1 | tee "$OUTDIR/curvature_run.log"

echo ""
echo "=== STRUCTURE DISCOVERY ==="
python "$GAM/examples/olmo_structure_discovery.py" 2>&1 | tee "$OUTDIR/structure_discovery_run.log"

echo ""
echo "=== LAYER TRANSPORT ==="
python "$GAM/examples/olmo_layer_transport.py" 2>&1 | tee "$OUTDIR/layer_transport_run.log"

echo ""
echo "=== ALL DONE ==="
ls -lh "$OUTDIR/"*.png "$OUTDIR/"*.json 2>/dev/null || true

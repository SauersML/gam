#!/usr/bin/env bash
set -Eeuo pipefail
helper="$HOME/mpd-data/codex/targeted-build-20261004/build.sh"
bash "$helper" 34d2523394879b9784dede47b8bfcbd6ad4eea3b mpd_shared_geometry_evaluate_2951
python "$HOME/mpd-data/bench/codex-shared-geometry-fit/bank-34d2/prepare.py"
bash "$helper" 34d2523394879b9784dede47b8bfcbd6ad4eea3b mpd_candidate_frontier_2951
bash "$helper" 34d2523394879b9784dede47b8bfcbd6ad4eea3b mpd_affine_mlp_fit_2951

#!/usr/bin/env bash
# Graph oracle e2e work on a GitHub Actions runner (#2951): public inputs only (VPD's vpd4l weights as our
# export, the template behaviors, VPD's subcomponents), fetched from the rp-run-data release by sha256 as
# listed in gha_inputs.tsv (gha_inputs.py writes it), then SCRIPT ARGS... runs with every "@D@" in ARGS
# replaced by the fetched tree (the Mac's ~/mpd-data layout) and the checker at $MPD_BIN.
#   GHA_EXAMPLES=gam-mpd:mpd_graph_2951 bench/runpod/gha-run search-induction -- \
#     'bash bench/oracle/graph/e2e/gha_e2e.sh search.py @D@/graph_oracle/behaviors/vpd4l/induction_random.words8.json \
#        --export @D@/engine/vpd4l --objective shared --out $OUT/search'
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
A=https://github.com/SauersML/gam/releases/download/rp-run-data
D=${RUNNER_TEMP:-/tmp}/mpd-data
python3 -m pip install -q --user numpy
mkdir -p "$D"
# Only the inputs ARGS name (a path prefix under @D@) are fetched.
want=$(printf '%s\n' "$@" | grep -o '@D@/[^ ]*' | sed 's|@D@/||' || true)
while IFS=$'\t' read -r sha enc rel; do
    match=0
    for w in $want; do case $rel in "$w"|"$w"/*) match=1 ;; esac; done
    [ $match = 1 ] || continue
    [ -f "$D/$rel" ] && continue
    mkdir -p "$(dirname "$D/$rel")"
    name=$sha
    [ "$enc" = raw ] || name=$sha.$enc
    curl -sSfL --retry 3 -o "$D/$rel.part" "$A/$name"
    if [ "$enc" = raw ]; then
        mv "$D/$rel.part" "$D/$rel"
    else  # an f32 or u16 encoding of a float64 file: widened back to the same bytes
        python3 -c 'import sys, numpy as np
np.fromfile(sys.argv[1], dtype="<f4" if sys.argv[2] == "f32" else "<u2").astype("<f8").tofile(sys.argv[3])' "$D/$rel.part" "$enc" "$D/$rel"
        rm "$D/$rel.part"
    fi
    echo "$sha  $D/$rel" | sha256sum -c --quiet
done < "$here/gha_inputs.tsv"
script=$1
shift
args=()
for a in "$@"; do args+=("${a//@D@/$D}"); done
export GRAPH_CHECKER=$MPD_BIN/mpd_graph_2951 MEM_LEASE_GIB=1 MPD_MEM_GIB=1 RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-4}
exec python3 "$here/$script" "${args[@]}"

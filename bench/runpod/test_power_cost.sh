#!/usr/bin/env bash
# rp-run keeps a power-capped GPU while its cost per token beats the next offer's (#2951): a card
# capped at fraction p of its default power runs at most about p of its measured speed, so
# unit_risk prices it at c / p (c its offer's cost: here the price, as for a run that names its
# cards). Takes rp-run's own gpu_power, next_offer and unit_risk, a probe log as the pod writes it
# (utilization, MiB free, power limit, default power, MiB total) and an attempts record with no rows
# (every failure probability the cloud's prior). Cases: a 4090 at $0.34/h held at 153 W of 450 W
# (34%, the host that ran 5x slower) against a secure 4090 at $0.74/h is discarded; a 3090 at
# $0.22/h held at 330 W of 420 W (79%) against a community 4090 at $0.34/h is kept.
#
#   bash bench/runpod/test_power_cost.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dir=$(mktemp -d); trap 'rm -rf "$dir"' EXIT
for f in gpu_power next_offer unit_risk; do sed -n "/^$f() {/,/^}/p" "$here/rp-run"; done > "$dir/functions.sh"
grep -q '^unit_risk() {' "$dir/functions.sh" && grep -q '^gpu_power() {' "$dir/functions.sh" || { echo "FAIL: functions not found in rp-run"; exit 1; }
source "$dir/functions.sh"
ssh_wait() { echo 60; }
on_pod() { cat "$probe"; }
CAP_SECONDS=1440 WS=/workspace OFFER_PER=''
ATTEMPTS=$dir/attempts.tsv
printf 'date\trun\tpod\tgpu\tcloud\thost\tdatacenter\tprice\tseconds\toutcome\tdetail\taddress\n' > "$ATTEMPTS"
declare -A spent=()
fail=0
probe=$dir/probe34
for i in 1 2 3 4 5; do echo "0, 24080, 153.00, 450.00, 24564"; done > "$probe"
p=$(gpu_power)
OFFERS=$'COMMUNITY|0.34|NVIDIA GeForce RTX 4090\nSECURE:cheapest-per-token|0.74|NVIDIA GeForce RTX 4090'
risk=$(unit_risk h1 - COMMUNITY 0.34 "NVIDIA GeForce RTX 4090" - "$p")
[ -n "$risk" ] || { echo "FAIL: a 4090 at $p of its power is kept"; fail=1; }
probe=$dir/probe79
for i in 1 2 3 4 5; do echo "0, 24080, 330.00, 420.00, 24564"; done > "$probe"
p2=$(gpu_power)
OFFERS=$'COMMUNITY|0.22|NVIDIA GeForce RTX 3090\nCOMMUNITY|0.34|NVIDIA GeForce RTX 4090'
risk2=$(unit_risk h2 - COMMUNITY 0.22 "NVIDIA GeForce RTX 3090" - "$p2")
[ -z "$risk2" ] || { echo "FAIL: a 3090 at $p2 of its power is discarded: $risk2"; fail=1; }
probe=$dir/probe100
for i in 1 2 3 4 5; do echo "0, 24080, 450.00, 450.00, 24564"; done > "$probe"
[ "$(gpu_power)" = 1 ] || { echo "FAIL: an uncapped card reads $(gpu_power)"; fail=1; }
(( fail == 0 )) && echo "PASS: the 4090 at $p of its power is discarded ($risk); the 3090 at $p2 is kept; an uncapped card reads 1"
exit $fail

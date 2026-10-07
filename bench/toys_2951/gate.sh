#!/bin/zsh
# The #2951 toy gate, end to end: a regression check every change to the fitting method must pass,
# on toys whose mechanisms are known (never results). Every toy, real-valued ones included, goes
# through the same fitter (mpd_library_mdl_2951 with library_vpd).
#
# usage: bench/toys_2951/gate.sh ROOT BINARY [EPOCHS] [TOY...]
#   ROOT    where the toys live (trained here when missing), e.g. ~/mpd-data/toys_gate
#   BINARY  mpd_library_mdl_2951 built at the commit under test (fastcheck build --release)
#   TOY     the toys to run (all seven when none), each also as seeds 1 and 2 (TOY_s1, TOY_s2)
#
# Per toy and seed: train it if missing (train_toys.py); score the native and truth references
# (score_toys.py --references); write the three start arms (toy_start.py); the engine's edits on
# the native start (gaps 0 up to rounding; its manifest is every fit's); then four fits: each arm
# at the toy's true per-token count (gate_settings.py BUDGET true) and grouped_own without a
# budget, each followed by its parts dump, its harness score and its engine edits. Then the table
# (table_toys.py) and universality across the seeds (universality.py).
set -e
root=$1; binary=$2; epochs=${3:-20}; shift 3 2>/dev/null || shift $#
toys=("$@"); (( ${#toys} )) || toys=(tms_40_10 tms_40_10_id resid_mlp_1l resid_mlp_2l resid_mlp_3l modadd_113 induction)
here=${0:A:h}
py=(env MPD_MEM_GIB=1 ~/mpd-data/venv/bin/python)
fits=(per_slice_own:true grouped_own:true grouped_direction:true grouped_own:none)
for toy in $toys; do
  for seed in 0 1 2; do
    name=$toy; (( seed )) && name=${toy}_s$seed
    t=$root/$name
    [[ -f $t/truth.json ]] || mem-lease 4 ~/mpd-data/venv/bin/python $here/train_toys.py $toy@s$seed $root
    env MPD_MEM_GIB=3 ~/mpd-data/venv/bin/python $here/score_toys.py $t --references $root/refs/$name > /dev/null
    [[ -f $root/start/$name/start.json ]] || $py $here/toy_start.py $t $root/start/$name
    native=$root/engine/${name}_native
    [[ -f $native/EDITS_native.json ]] || $py $here/engine_edits.py $binary $t $native > /dev/null
    for fit in $fits; do
      arm=${fit%%:*}; budget=${fit#*:}
      out=$root/fit/$name.$arm.$budget
      mkdir -p $root/fit
      $py $here/gate_settings.py $t $root/start/$name $arm $budget $epochs $out.json
      [[ -f $out/REPORT.json ]] || mem-lease 4 $binary $t $out.json $out host > $out.log 2>&1
      $binary $t $out.json $out host parts $out/parts >> $out.log 2>&1
      env MPD_MEM_GIB=3 ~/mpd-data/venv/bin/python $here/score_toys.py $t $out/parts > /dev/null
      $py $here/engine_edits.py $binary $t $out $out.json checkpoint.bin $native/MANIFEST_native.json > /dev/null
    done
  done
done
$py $here/table_toys.py $root/refs/*/*/SCORE_*.json $root/fit/*/parts/SCORE_*.json $root/engine/*/SCORE_*_edits.json $root/fit/*/SCORE_*_edits.json
for toy in $toys; do
  for fit in $fits; do
    arm=${fit%%:*}; budget=${fit#*:}
    args=()
    for name in $toy ${toy}_s1 ${toy}_s2; do args+=($root/$name $root/fit/$name.$arm.$budget/parts); done
    echo "universality $toy $arm $budget"
    env MPD_MEM_GIB=3 ~/mpd-data/venv/bin/python $here/universality.py $args
  done
done

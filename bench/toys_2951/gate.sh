#!/bin/zsh
# The #2951 toy gate, end to end: a regression check every change to the fitting method must pass,
# on toys whose mechanisms are known (never results). Every toy, real-valued ones included, goes
# through the same fitter (mpd_library_mdl_2951 with library_vpd).
#
# usage: bench/toys_2951/gate.sh ROOT BINARY [EPOCHS] [TOY...]
#   ROOT    where the toys live (trained here when missing), e.g. ~/mpd-data/toys_gate
#   BINARY  mpd_library_mdl_2951 built at the commit under test (fastcheck build --release)
#   TOY     the toys to run (all seven when none), each also as seeds 1 and 2 (TOY_s1, TOY_s2)
# GATE_DEVICE (host or gpu, default host) is the fits' device; GATE_FITS (default the five below,
# START:ARM:BUDGET separated by spaces) runs only those fits, so lanes can split a toy's fits; GATE_LEASE
# (GiB, default 4) is a fit's memory lease. A fit that fails (it diverges, or outgrows its lease) is
# reported in FAILED.txt and the gate goes on to the next.
#
# Per toy and seed: train it if missing (train_toys.py); score the native and truth references
# (score_toys.py --references); write the starts (the driver's frame-start: the tight and
# dictionary frames and the per-head/per-neuron baseline, gam_mpd::library_frame); the engine's
# edits on the native start (gaps 0 up to rounding; its manifest is every fit's); then the fits,
# START:ARM:BUDGET (BUDGET none, or true: the toy's true per-token count, a labelled diagnostic),
# each followed by its parts dump, its harness score and its engine edits. Then the table
# (table_toys.py) and universality across the seeds (universality.py).
set -e
root=$1; binary=$2; epochs=${3:-20}; shift 3 2>/dev/null || shift $#
toys=("$@"); (( ${#toys} )) || toys=(tms_40_10 tms_40_10_id resid_mlp_1l resid_mlp_2l resid_mlp_3l modadd_113 induction)
here=${0:A:h}
py=(env MPD_MEM_GIB=1 ~/mpd-data/venv/bin/python)
fits=(${=GATE_FITS:-dictionary:frame_own:none dictionary:frame_own:true tight:frame_own:none heads_neurons:grouped_own:none heads_neurons:grouped_direction:none})
device=${GATE_DEVICE:-host}
for toy in $toys; do
  for seed in 0 1 2; do
    name=$toy; (( seed )) && name=${toy}_s$seed
    t=$root/$name
    [[ -f $t/truth.json ]] || mem-lease 4 ~/mpd-data/venv/bin/python $here/train_toys.py $toy@s$seed $root
    env MPD_MEM_GIB=3 ~/mpd-data/venv/bin/python $here/score_toys.py $t --references $root/refs/$name > /dev/null
    if [[ ! -f $root/start/$name/dictionary/start.json ]]; then
      $py $here/gate_settings.py $t $root/start/$name/dictionary frame_own none $epochs $root/start/$name.json > /dev/null
      $binary $t $root/start/$name.json $root/start/$name.out host frame-start $root/start/$name > $root/start/$name.log 2>&1
    fi
    native=$root/engine/${name}_native
    [[ -f $native/EDITS_native.json ]] || $py $here/engine_edits.py $binary $t $native > /dev/null
    for fit in $fits; do
      start=${fit%%:*}; rest=${fit#*:}; arm=${rest%%:*}; budget=${rest#*:}
      out=$root/fit/$name.$start.$arm.$budget
      mkdir -p $root/fit
      $py $here/gate_settings.py $t $root/start/$name/$start $arm $budget $epochs $out.json
      if ! { [[ -f $out/REPORT.json ]] || mem-lease ${GATE_LEASE:-4} $binary $t $out.json $out $device > $out.log 2>&1; }; then
        echo "$name.$start.$arm.$budget: $(tail -1 $out.log)" >> $root/FAILED.txt
        continue
      fi
      if ! { $binary $t $out.json $out host parts $out/parts >> $out.log 2>&1 && env MPD_MEM_GIB=3 ~/mpd-data/venv/bin/python $here/score_toys.py $t $out/parts > /dev/null; }; then
        echo "$name.$start.$arm.$budget parts: $(tail -1 $out.log)" >> $root/FAILED.txt
        continue
      fi
      $py $here/engine_edits.py $binary $t $out $out.json checkpoint.bin $native/MANIFEST_native.json > /dev/null || echo "$name.$start.$arm.$budget edits: failed" >> $root/FAILED.txt
    done
  done
done
$py $here/table_toys.py $root/refs/*/*/SCORE_*.json $root/fit/*/parts/SCORE_*.json $root/engine/*/SCORE_*_edits.json $root/fit/*/SCORE_*_edits.json
for toy in $toys; do
  for fit in $fits; do
    start=${fit%%:*}; rest=${fit#*:}; arm=${rest%%:*}; budget=${rest#*:}
    args=()
    for name in $toy ${toy}_s1 ${toy}_s2; do [[ -f $root/fit/$name.$start.$arm.$budget/parts/parts.json ]] && args+=($root/$name $root/fit/$name.$start.$arm.$budget/parts); done
    (( ${#args} >= 4 )) || continue
    echo "universality $toy $start $arm $budget"
    env MPD_MEM_GIB=3 ~/mpd-data/venv/bin/python $here/universality.py $args
  done
done

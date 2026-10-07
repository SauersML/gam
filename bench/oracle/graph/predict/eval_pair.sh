#!/usr/bin/env bash
# eval_kl.py for the base oracle and the trained one on the same held-out questions (#2951):
#   eval_pair.sh ORACLE_DIR 'HELDOUT_GLOB' ADAPTERS OUT_DIR
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$HOME/oracle-venv/bin/python}
[ -x "$PY" ] || PY=python3
mkdir -p "$4"
$PY "$here/eval_kl.py" --model "$1" --heldout "$2" --out "$4/eval_kl_base.json"
$PY "$here/eval_kl.py" --model "$1" --adapters "$3" --heldout "$2" --out "$4/eval_kl_trained.json"

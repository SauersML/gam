"""by_effect table from EDITS_*.json (edits driver at 9b43f79505): per arm and family, per bin of the
edit's effect on M at the edited token (KL(M_e || M), bits): experiments, the gap KL(M_e || P_e) over
the scored tokens and at the edited token.
usage: by_effect_table.py NAME=EDITS.json ..."""
import json, sys
for arg in sys.argv[1:]:
    name, path = arg.split('=', 1)
    try:
        d = json.load(open(path))
    except Exception as e:
        print(f'{name}: missing ({e})'); continue
    print(f'== {name} ({d["parts"]} parts, {d["sequences"]}, {d["edits_per_sequence"]} edits/sequence)')
    for fam, v in sorted(d['families'].items(), key=lambda kv: (kv[1].get('objects', ''), kv[0])):
        print(f'  {fam:18s} [{v.get("objects", "?")}] all: n {v["experiments"]:4d} gap {v["mean_bits_per_token"]:.3f} at t {v["edited_token_mean_bits"]:.3f} effect at t {v["effect_edited_token_mean_bits"]:.3f}')
        for b in v.get('by_effect', []):
            lo, hi = b['effect_bits_at_edited_token']
            print(f'      effect [{lo}, {hi}): n {b["experiments"]:4d} gap {b["mean_bits_per_token"]:.3f} at t {b["edited_token_mean_bits"]:.3f} effect {b["effect_edited_token_mean_bits"]:.3f}')

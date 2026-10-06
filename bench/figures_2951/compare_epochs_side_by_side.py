"""Per epoch, two vpd4l transcoder fits side by side (the old prior grouping and the threshold group,
75a2712a6c): held-out F, KL at the posterior mean, active features per token per layer (from each
fit's checkpoint.json), and the median threshold ratio c/|g| per layer of each saved checkpoint
(checkpoint.bin, its epoch named in its header).

usage: epochs_side_by_side.py OLD_OUT THR_OUT [OUT.json]"""
import json, struct, sys, os
import numpy as np


def trajectory(out):
    p = f'{out}/checkpoint.json'
    if not os.path.exists(p):
        return {}
    h = json.load(open(p))
    rows = {}
    for name, ho in [('start', h.get('start'))] + [(f"epoch {e['epoch']}", e['held_out']) for e in h['epochs']]:
        if ho:
            rows[name] = {'F': ho['objective_bits_per_token'], 'kl_mean': ho['mean_bits_per_token'],
                          'active': [round(l['nonzero_per_token'], 1) for l in ho['layers'] if l['functions']]}
    return rows


def thresholds(path):
    """Per layer the median c/|g| of a checkpoint's posterior means: the gate [k, d] and gate bias
    [k, 1] operators found by shape, in order."""
    f = open(path, 'rb')
    n = struct.unpack('<Q', f.read(8))[0]
    h = json.loads(f.read(n))
    size = {'F32': 4, 'F64': 8, 'Bf16': 2}
    means = []
    for (r, c) in h['shapes']:
        first = None
        for j, p in enumerate(h['precision']):
            raw = f.read(r * c * size[p])
            if j == 0:
                dt = {'F32': '<f4', 'F64': '<f8'}.get(p)
                first = np.frombuffer(raw, dt).reshape(r, c).astype(np.float64) if dt else (np.frombuffer(raw, '<u2').astype(np.uint32) << 16).view(np.float32).reshape(r, c).astype(np.float64)
        means.append(first)
    shapes = [tuple(s) for s in h['shapes']]
    out = []
    for i in range(len(shapes) - 1):
        (k, d), (k2, one) = shapes[i], shapes[i + 1]
        if k == k2 and one == 1 and d > 1 and k > 1:
            out.append(round(float(np.median(means[i + 1][:, 0] / np.linalg.norm(means[i], axis=1))), 3))
    return h['epoch'], out


old, thr = sys.argv[1], sys.argv[2]
res = {'old': trajectory(old), 'thr': trajectory(thr)}
for tag, out in (('old', old), ('thr', thr)):
    for ck in ('checkpoint.start.bin', 'checkpoint.bin'):
        p = f'{out}/{ck}'
        if os.path.exists(p):
            e, r = thresholds(p)
            res.setdefault(f'{tag}_thresholds', {})[f'{ck} (next epoch {e})'] = r
keys = sorted(set(res['old']) | set(res['thr']), key=lambda k: (k != 'start', k))
for k in keys:
    o, t = res['old'].get(k), res['thr'].get(k)
    fmt = lambda r: f"F {r['F']:.3f} KL {r['kl_mean']:.3f} active {r['active']}" if r else '-'
    print(f'{k:8s} | old: {fmt(o)} | thr: {fmt(t)}')
for k in ('old_thresholds', 'thr_thresholds'):
    print(k, res.get(k))
if len(sys.argv) > 3:
    json.dump(res, open(sys.argv[3], 'w'), indent=1)

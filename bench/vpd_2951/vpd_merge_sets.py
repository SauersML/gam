"""Join the masked driver's per-passage set dumps into one stem (#2951), so the passages a job
array selected one at a time are scored together (`vpd_stepA_honesty.py`, which shares VPD's
adversary over every passage it reads).

Each input stem has `{stem}.{indptr,indices,offsets}.npy` (CSR over its positions, pieces numbered
site after site); every input must number the same library (equal offsets). The output stem gets
them in the order given.

usage: vpd_merge_sets.py OUT_STEM IN_STEM...
"""

import sys

import numpy as np

out, stems = sys.argv[1], sys.argv[2:]
indptr, indices, offsets, total = [np.zeros(1, dtype=np.int64)], [], None, 0
for stem in stems:
    ip, ix, off = (np.load(f"{stem}.{k}.npy") for k in ("indptr", "indices", "offsets"))
    if offsets is None:
        offsets = off
    elif not np.array_equal(off, offsets):
        raise SystemExit(f"{stem}: another library's numbering")
    indptr.append(ip[1:] + total)
    indices.append(ix)
    total += int(ip[-1])
for name, values in (("indptr", np.concatenate(indptr)), ("indices", np.concatenate(indices)), ("offsets", offsets)):
    np.save(f"{out}.{name}.npy", values.astype("<i8"))
print(f"{out}: {len(stems)} stems, {len(indptr[0]) - 1 + sum(len(p) for p in indptr[1:])} positions, {total} active")

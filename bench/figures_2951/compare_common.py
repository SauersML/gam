"""Shared helpers of the #2951 head-to-head figures (compare_frontier.py, compare_f_edits.py)."""
import functools, hashlib, json, os, re


@functools.lru_cache(maxsize=None)
def experiments_digest(path):
    """The SHA-256 of an immutable manifest's experiments: two manifests listing the same experiments
    (one with native weight edits drawn beside them) score the same shared operations."""
    return hashlib.sha256(json.dumps(json.load(open(path))['experiments'], sort_keys=True).encode()).hexdigest()


def experiments_of(r):
    """The experiments an EDITS file (`r`) was scored on: its immutable manifest's experiments digest
    (a pod's relative path ../mpd-data/... read from ~/mpd-data), or the manifest file's SHA-256 where
    the file is not here; none without a manifest."""
    m = (r or {}).get('manifest') or {}
    path = re.sub(r'^(\.\./)+mpd-data/', '/Users/user/mpd-data/', m.get('file') or '')
    if path and os.path.exists(path):
        return experiments_digest(path)
    return m.get('sha256')


def latest_scores(d, arm):
    """An arm's EDITS file in directory `d`: on the weight-edit manifest (s2, the s1 experiments plus
    native weight edits) where it exists, else on s1."""
    return next((f for f in (f'EDITS_{arm}_s2.json', f'EDITS_{arm}_m1.json') if os.path.exists(f'{d}/{f}')), f'EDITS_{arm}_m1.json')

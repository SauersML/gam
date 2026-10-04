import json,hashlib,subprocess,os
from pathlib import Path
root=Path.home()/'mpd-data';stage=root/'bench/codex-shared-geometry-fit/bank-34d2';pin='34d2523394879b9784dede47b8bfcbd6ad4eea3b';d=Path.home()/'mpd-bin-targeted'/pin[:12];b=d/'mpd_candidate_frontier_2951';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert(d/'COMMIT').read_text().strip()==pin;assert sha(b)==(d/'mpd_candidate_frontier_2951.sha256').read_text().split()[0]
parent=root/'cluster/codex-shared-geometry-bank-34d2';bank=parent/'BANK.json';hashes=json.loads((parent/'PREPARED_SHA256.json').read_text());entries=json.loads(bank.read_text());assert len(entries)==4
for e in entries:assert sha(Path(e['artifact']))==hashes[e['label']]
spec=root/'codex/copy-whole-sequence-validation-20261004/spec.json';assert sha(spec)=='3c05b66324dfb02e33da7eff98784a7ec432123e24fea78018b892efb4256436'
out=parent/'evaluation';assert not out.exists();cmd=[str(b),str(root/'engine/vpd4l_frontier32'),str(spec),str(bank),str(out)]+json.loads((stage/'FRONTIER_OPTIONS.json').read_text())
(parent/'EVALUATION_EXECUTION.json').write_text(json.dumps({'source':pin,'job':os.environ.get('SLURM_JOB_ID'),'binary_sha256':sha(b),'bank_sha256':sha(bank),'command':cmd},indent=2));subprocess.run(cmd,check=True)
r=json.loads((out/'report.json').read_text());assert r['distinct_candidates']==5 and r['measured_candidates']==5

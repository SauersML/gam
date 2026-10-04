import json,hashlib,subprocess,os
from pathlib import Path
here=Path.home()/'mpd-data/codex/resident-rule-fit-gpu-control-20261004/retry-58c2'
p=json.loads((here/'PROTOCOL.json').read_text());d=Path.home()/'mpd-bin-targeted'/p['source_commit'][:12];b=d/p['example']
assert (d/'COMMIT').read_text().strip()==p['source_commit']
sha=hashlib.sha256(b.read_bytes()).hexdigest();assert sha==(d/(p['example']+'.sha256')).read_text().split()[0]
out=Path.home()/'mpd-data/cluster/codex-resident-rule-fit-gpu-58c2';assert not out.exists();out.mkdir()
r=subprocess.run([str(b)],text=True,capture_output=True)
(out/'stdout.txt').write_text(r.stdout);(out/'stderr.txt').write_text(r.stderr)
(out/'PROVENANCE.json').write_text(json.dumps({'protocol':p,'binary_sha256':sha,'job':os.environ.get('SLURM_JOB_ID'),'returncode':r.returncode},indent=2))
print(r.stdout);print(r.stderr);r.check_returncode()
(out/'report.json').write_text(json.dumps(json.loads(r.stdout),indent=2)+'\n')

import json,hashlib,subprocess,sys,os
from pathlib import Path
root=Path.home()/'mpd-data';here=root/'codex/language-resident-metrics-20261004';p=json.loads((here/'PROTOCOL.json').read_text());mode=sys.argv[1]
assert mode in ('smoke','full')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
assert sha(root/p['export']/'tokens.f64')==p['tokens_sha256']
assert sha(root/p['parent_spec'])==p['parent_spec_sha256']
spec=root/(p['smoke_spec'] if mode=='smoke' else p['parent_spec'])
assert sha(spec)==p['smoke_spec_sha256' if mode=='smoke' else 'parent_spec_sha256']
for entry in p['artifacts']:assert sha(root/entry['file'])==entry['sha256']
baseline=root/p['normalized_baselines'][mode]['file'];assert sha(baseline)==p['normalized_baselines'][mode]['sha256']
assert json.loads(baseline.read_text())['passes']
raw_baseline=root/p['raw_baselines'][mode]['file'];assert sha(raw_baseline)==p['raw_baselines'][mode]['sha256'];assert json.loads(raw_baseline.read_text())['passes']
bin_dir=Path.home()/'mpd-bin-targeted'/p['source_commit'][:12];binary=bin_dir/p['example']
assert (bin_dir/'COMMIT').read_text().strip()==p['source_commit'];assert sha(binary)==(bin_dir/(p['example']+'.sha256')).read_text().split()[0]
if mode=='full':
 gate_dir=root/'cluster'/('codex-language-resident-smoke-'+p['source_commit'][:4]);gate=json.loads((gate_dir/'report.json').read_text());assert gate['passes']
 assert json.loads((gate_dir/'PROVENANCE.json').read_text())['protocol']['source_commit']==p['source_commit']
out=root/'cluster'/('codex-language-resident-'+mode+'-'+p['source_commit'][:4]);out.mkdir(exist_ok=True);report=out/'report.json';assert not report.exists(),'freshoutputrequired'
cmd=[str(binary),str(root/p['export']),str(spec),str(report),str(baseline),str(raw_baseline),'yes' if mode=='smoke' else 'no']+[str(root/e['file']) for e in p['artifacts']]
(out/'PROVENANCE.json').write_text(json.dumps({'protocol':p,'protocol_sha256':sha(here/'PROTOCOL.json'),'script_sha256':sha(Path(__file__)),'binary_sha256':sha(binary),'job':os.environ.get('SLURM_JOB_ID'),'mode':mode,'command':cmd},indent=2)+'\n')
subprocess.run(cmd,check=True)
r=json.loads(report.read_text());assert r['passes'] and r['episode_count']==(2 if mode=='smoke' else 80)
print('PASS resident raw-logit production metric '+mode+'; defaultacceptance unchanged',flush=True)

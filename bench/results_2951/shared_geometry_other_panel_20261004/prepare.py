import json,hashlib,subprocess,os
from pathlib import Path
root=Path.home()/'mpd-data';stage=root/'bench/codex-shared-geometry-fit/bank-34d2';pin='34d2523394879b9784dede47b8bfcbd6ad4eea3b';d=Path.home()/'mpd-bin-targeted'/pin[:12];b=d/'mpd_shared_geometry_evaluate_2951';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert (d/'COMMIT').read_text().strip()==pin
parent=root/'cluster/codex-shared-geometry-bank-34d2';parent.mkdir(exist_ok=True);bank=[]
for arm,poolsha in json.loads((stage/'POOLS.json').read_text()).items():
 pool=root/('cluster/codex-shared-geometry-256-'+arm+'-f822/measurement/fitted-pool.artifact');assert sha(pool)==poolsha
 out=parent/('prepared-'+arm);assert not out.exists()
 cmd=[str(b),str(root/'engine/vpd4l_frontier32'),str(pool),poolsha,str(root/('bench/codex-shared-geometry-fit/controls-256-f822/'+arm+'.json')),str(root/'codex/copy-whole-sequence-validation-20261004/spec.json'),'3c05b66324dfb02e33da7eff98784a7ec432123e24fea78018b892efb4256436',str(out),str(stage/'BUDGET.json'),'prepare_only']
 subprocess.run(cmd,check=True);artifact=out/'candidate.artifact';bank.append({'label':'shared-geometry256-'+arm,'artifact':str(artifact)})
(parent/'BANK.json').write_text(json.dumps(bank,indent=2)+'\n');(parent/'PREPARED_SHA256.json').write_text(json.dumps({e['label']:sha(Path(e['artifact'])) for e in bank},indent=2)+'\n')

import json,hashlib,subprocess,os,time,sys
from pathlib import Path
root=Path.home()/'mpd-data';stage=root/'bench/codex-shared-geometry-fit/paired-continuation-f56b';arm=sys.argv[1];assert arm in ['learned','frozen_native'];pin='f56bf43964d193f56cda1bbfeb6182619bd39c7c';name='mpd_shared_geometry_fit_2951'
config=stage/(arm+'.json');p=json.loads(config.read_text());export=root/'engine/vpd4l_frontier32';extract=root/'cluster/codex-affine-mlp-9c504de18084/extract/EXTRACT.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(extract)==p['extract_sha256'];assert sha(export/'export.json')==p['export_json_sha256']
d=Path.home()/'mpd-bin-targeted'/pin[:12];b=d/name;assert(d/'COMMIT').read_text().strip()==pin;assert sha(b)==(d/(name+'.sha256')).read_text().split()[0]
parent=root/('cluster/codex-shared-geometry-continuation-'+arm+'-f56b');parent.mkdir(exist_ok=True);out=parent/'measurement';assert not out.exists()
cmd=[str(b),str(export),str(extract),str(config),str(out),'cuda']
(parent/'PROVENANCE.json').write_text(json.dumps({'source_commit':pin,'job':os.environ.get('SLURM_JOB_ID'),'command':cmd,'config_sha256':sha(config),'binary_sha256':sha(b),'aggregate_fit_cap_gpu_seconds':3600},indent=2))
t=time.monotonic();subprocess.run(cmd,check=True);(parent/'WALL.json').write_text(json.dumps({'wall_seconds':time.monotonic()-t,'scope':'setup+fit+saved replay; optimizer separately inREPORT'},indent=2))

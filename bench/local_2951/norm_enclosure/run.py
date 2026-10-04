"""Pinned bounded real-panel comparison; no fitting or fidelity-bank expansion."""
import json,pathlib,hashlib,subprocess,time,os,resource
root=pathlib.Path.home()/'mpd-data';stage=root/'codex/local-norm-enclosure-20261004';p=json.loads((stage/'PROTOCOL.json').read_text());pin=p['source_commit'];directory=pathlib.Path.home()/'mpd-bin-targeted'/pin[:12];exe=directory/'mpd_local_device_probe_2951'
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
if (directory/'COMMIT').read_text().strip()!=pin or sha(exe)!=(directory/(exe.name+'.sha256')).read_text().split()[0]:raise RuntimeError('binary source/hash mismatch')
if sha(root/'engine/vpd4l_frontier32/export.json')!=p['export_sha256']:raise RuntimeError('export lineage changed')
out=root/'cluster'/p['output']/'measurement';out.mkdir(parents=True,exist_ok=False);entries=[]
for artifact in p['artifacts']:
 file=root/artifact['path']
 if sha(file)!=artifact['sha256']:raise RuntimeError('saved artifact changed')
 entries.append({'label':artifact['label'],'artifact':str(file)})
bank=out/'BANK.json';bank.write_text(json.dumps(entries,indent=2)+'\n');command=[str(exe),str(root/'engine/vpd4l_frontier32'),str(bank),str(out/'REPORT.json')]+[k+'='+str(p[k]) for k in ['sequences','context','batch','trace_bytes','source_bytes']]+['deltas='+','.join(map(str,p['deltas']))]
(out/'PROVENANCE.json').write_text(json.dumps({'source_commit':pin,'protocol_sha256':sha(stage/'PROTOCOL.json'),'binary_sha256':sha(exe),'job':os.environ.get('SLURM_JOB_ID'),'command':command,'IEEE_RN_gradual_underflow':'assumed, not runtime verified'},indent=2)+'\n');begun=time.monotonic()
with (out/'probe.stdout').open('x') as stdout,(out/'probe.stderr').open('x') as stderr:
 result=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=p['wall_seconds'])
state={'returncode':result.returncode,'seconds':time.monotonic()-begun,'peak_child_rss_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024}
if result.returncode:
 (out/'STATE.json').write_text(json.dumps(state,indent=2)+'\n');raise RuntimeError('probe failed; unresolved; see stderr')
r=json.loads((out/'REPORT.json').read_text());prior=json.loads((root/'cluster/codex-local-resident-norms/report.json').read_text());changes=[]
if len(r['records'])!=2:raise RuntimeError('real panel inventory changed')
for current,old in zip(r['records'],prior['records']):
 if current['artifact_sha256']!=old['artifact_sha256']:raise RuntimeError('comparison artifacts changed')
 for pair,previous in zip(current['pairs'],old['pairs']):
  for grid,before in zip(pair['grid'],previous['grid']):
   if grid['delta']!=before['delta']:raise RuntimeError('declared grid changed')
   for backend in ['cpu','cuda']:
    if grid[backend]!=before[backend]:changes.append({'artifact':current['label'],'repeat':pair['repeat'],'delta':grid['delta'],'backend':backend,'before':before[backend],'after':grid[backend]})
  if not pair['resident_norms']['host_center_bits_equal'] or not pair['resident_norms']['sound_interval_overlap'] or not pair['resident_norms']['declared_grid_classifications_equal']:raise RuntimeError('host/resident parity failed')
state.update(state='complete',report_sha256=sha(out/'REPORT.json'),prior_report_sha256=sha(root/'cluster/codex-local-resident-norms/report.json'),classification_changes=changes,all_frozen_classifications_unchanged=not changes,protocol=p);(out/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')

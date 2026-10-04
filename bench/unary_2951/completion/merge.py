"""Complete exactly the one retained unknown; preserve original sweep evidence."""
import json,sys,hashlib
from pathlib import Path
old,new=map(Path,sys.argv[1:]); merged=json.loads((old/'MERGED.json').read_text()); report=json.loads((new/'evaluation/REPORT.json').read_text())
def read(p):return json.loads(p.read_text())
for filename in ['INVENTORY.json','RUN_SPEC.json','TOKEN_FAMILY.json']:
 a,b=read(old/'shard-2/evaluation'/filename),read(new/'evaluation'/filename)
 if filename=='RUN_SPEC.json':
  for scope in [a['scope'],b['scope']]:
   for key in ['assessment_candidate_index','max_assess','max_seconds','stop_when_all_cost_gaps_zero']:scope.pop(key,None)
 if a!=b:raise RuntimeError('scope changed '+filename)
for a,b in zip(merged['records'],report['records']):
 if (a['index'],a['candidate'],a.get('cost_bits'))!=(b['index'],b['candidate'],b.get('cost_bits')):raise RuntimeError('inventory/price changed')
measured=[r['index'] for r in report['records'] if r.get('quality_measured')]
if measured!=[0,3789]:raise RuntimeError('unexpected measured scope '+str(measured))
a,b=merged['records'][0],report['records'][0]
if a['states']!=b['states']:raise RuntimeError('native verdict changed')
for x,y in zip(a['local']['blocks'],b['local']['blocks']):
 if x['name']!=y['name'] or abs(x['worst']-y['worst'])>x['numerical_error']+y['numerical_error']:raise RuntimeError('native Local mismatch')
for x,y in zip(a['run']['groups'],b['run']['groups']):
 if x[0]!=y[0] or abs(x[1]-y[1])>x[2]+y[2]:raise RuntimeError('native Run mismatch')
request=read(new/'evaluation/REQUESTED_CANDIDATE.json')
if request['index']!=3789 or request['cost_bits']!=2139046636 or not request['ordinary_decode_canonical_coverage_cost']:raise RuntimeError('requested artifact mismatch')
if merged['records'][3789].get('quality_measured'):raise RuntimeError('old point already measured')
merged['records'][3789]=report['records'][3789]
for g,p in enumerate(merged['points']):
 lower=min(r.get('cost_bits') or 0 for r in merged['records'] if r['states'][g]!='Violates'); viable=[(r['cost_bits'],r['index']) for r in merged['records'] if r['states'][g]=='Verified'];best=min(viable) if viable else None
 p.update(lower_cost=lower,upper_cost=best[0] if best else None,selected=best[1] if best else None,gap=best[0]-lower if best else None)
merged.update(assessed=sum(bool(r.get('quality_measured')) for r in merged['records']),unmeasured=sum(not bool(r.get('quality_measured')) for r in merged['records']),completion_report=str(new/'evaluation/REPORT.json'),requested_saved_artifact=request,native_control_agreement='within recorded numeric bands',source_pin=read(new/'PROTOCOL.json')['source_commit'])
(old/'MERGED_COMPLETE.json').write_text(json.dumps(merged,indent=2)+'\n')
print(json.dumps({'assessed':merged['assessed'],'unmeasured':merged['unmeasured'],'points':merged['points'],'sha256':hashlib.sha256((old/'MERGED_COMPLETE.json').read_bytes()).hexdigest()},indent=2))

import json, numpy as np
B=json.load(open('blind_prompts.json')); names=[u['function'] for u in B]
F=json.load(open('out/functions.json')); col={f['name']:i for i,f in enumerate(F)}; cols=[col[n] for n in names]
R=json.load(open('ref/attributions.json'))
ref=np.concatenate([np.fromfile('ref/prompt%d.outputs.f64'%i,dtype='<f8').reshape(p['rows'],p['functions'])[:,cols] for i,p in enumerate(R['prompts'])])
p90=np.percentile(ref,90,axis=0)
idx=json.load(open('blind_index.json')); A=json.load(open('blind_out/attributions.json'))
per={}; others=[]; fails=[]
for i,(e,p) in enumerate(zip(idx,A['prompts'])):
    out=np.fromfile('blind_out/prompt%d.outputs.f64'%i,dtype='<f8').reshape(p['rows'],p['functions'])[-1,cols]
    j=names.index(e['function']); fired=out>p90
    ok=bool(fired[j]) if e['kind']=='fire' else not fired[j]
    per.setdefault(e['function'],[]).append((e['kind'],ok))
    if e['kind']=='fire': others.append(np.delete(fired,j).mean())
    else: others_q=None
    if not ok: fails.append((e['function'], e['kind'], e['text'][-50:], round(float((ref[:,j]<out[j]).mean()),2)))
allr=[o for v in per.values() for _,o in v]
print('blind: correct %d / %d = %.3f'%(sum(allr),len(allr),np.mean(allr)))
for k in ('fire','quiet'):
    x=[o for v in per.values() for kk,o in v if kk==k]; print(k, '%.3f'%np.mean(x))
print('functions with all 4 right: %d / %d'%(sum(all(o for _,o in v) for v in per.values()), len(per)))
for tag in ('.H','.M'):
    x=[o for n,v in per.items() if tag in n for _,o in v]; print(tag, '%d/%d'%(sum(x),len(x)))
print('fire prompts: other 49 functions also firing %.3f'%np.mean(others))
json.dump({'correct':float(np.mean(allr)),'failures':fails}, open('blind_results.json','w'), ensure_ascii=False, indent=1)
for f in fails: print(f)

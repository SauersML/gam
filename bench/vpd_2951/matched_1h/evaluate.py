"""Existing nano VPD baseline checkpoint conversion and frozen CUDA evaluation.
All Torch math is the upstream baseline/diagnostic, not our rule discovery method.
No model states from native evaluation enter the autonomous selection policy.
"""
import time
clock=time.monotonic()
import os,sys,json,hashlib,importlib.util,resource
from pathlib import Path
os.environ['WANDB_MODE']='disabled'
os.environ['WANDB_DISABLED']='true'
import numpy as np
import torch
import torch.nn.functional as F
stage=Path(__file__).resolve().parent
protocol=json.loads((stage/'PROTOCOL.json').read_text())
root=Path.home()/'mpd-data/cluster'/protocol['output']
root.mkdir(parents=True,exist_ok=True)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def load_module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
source=Path.home()/protocol['source_dir_relative']
for name,digest in protocol['source_sha256'].items():
 if sha(source/name)!=digest:raise RuntimeError('pinned baseline source changed '+name)
vm=load_module(source/'vpd_model.py','matched_eval_target')
nano=load_module(source/'run.py','matched_eval_nano')
if sha(stage/'spec.json')!=protocol['spec_sha256']:raise RuntimeError('frozen panel changed')
def nanokey(name):return 'sites.'+name.replace('.','-')
def conversion():
 checkpoint=Path.home()/protocol['checkpoint_relative']
 raw=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
 if raw['step']!=protocol['training_steps'] or not raw['finished'] or raw['smoke'] or raw['budget_hours']!=1:raise RuntimeError('final onehour checkpoint provenance mismatch')
 config_hash=hashlib.sha256(json.dumps(raw['config'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
 if config_hash!=protocol['config_sha256']:raise RuntimeError('training config changed')
 uv={};native={}
 for name in vm.site_names():
  prefix=nanokey(name)
  U,V=raw['target'][prefix+'.U'],raw['target'][prefix+'.V']
  if U.dtype!=torch.float32 or V.dtype!=torch.float32 or U.shape[0]!=raw['config']['C_per_module'][prefix] or U.shape[0]!=V.shape[1]:raise RuntimeError('component tensor layout/dtype mismatch')
  uv[name]={'U':U,'V':V};native[name]=raw['target'][prefix+'.W_target']
 ci=raw['ci'];fallback={k:v for k,v in raw['target'].items() if not k.startswith('sites.')}
 if any(not bool(torch.isfinite(v).all()) for v in list(ci.values())+list(native.values())+list(fallback.values())+[t for pair in uv.values() for t in pair.values()]):raise RuntimeError('checkpoint nonfinite: evaluation unresolved')
 # Native embeddings are tied and counted once. RoPE tables are deterministic
 # architecture computations, not fitted independent weights; structure unpaid here.
 fallback_names=[k for k in fallback if k=='wte' or k=='ln_f' or k.startswith('norms.')]
 counts={'components':sum(v.numel() for pair in uv.values() for v in pair.values()),'CI':sum(v.numel() for v in ci.values()),'native_embedding_norms':sum(fallback[k].numel() for k in fallback_names)}
 payload={'config':raw['config'],'uv':uv,'native':native,'fallback':fallback,'ci':ci,'measurement':raw['measurement'],'train_cursor':raw['train_cursor'],'eval_cursor':raw['eval_cursor']}
 dest=root/'baseline.slim.pt'
 if dest.exists():raise RuntimeError('fresh conversion required')
 torch.save(payload,dest)
 record={'raw_checkpoint_sha256':sha(checkpoint),'raw_checkpoint_bytes':checkpoint.stat().st_size,'slim_sha256':sha(dest),'slim_bytes':dest.stat().st_size,'config_sha256':config_hash,'training':raw['measurement'],'numeric_parameters':counts,'numeric_payload_bits_32':32*sum(counts.values()),'C32_total':None,'structural_cost_status':'unresolved: control iteration, threshold, architecture constants, wiring and bindings not encoded in current Artifact grammar','native_fallback_names':fallback_names,'no_optimizer_in_slim':True,'seconds':time.monotonic()-clock,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'source_sha256':protocol['source_sha256'],'evaluation_source_sha256':sha(Path(__file__))}
 save(root/'CONVERSION.json',record);print(json.dumps(record),flush=True)
class Adapter(torch.nn.Module):
 def __init__(self,body):super().__init__();self.body=body
 def forward(self,acts):
  _,_,pre=self.body({nanokey(n):v for n,v in acts.items()})
  return {n:pre[nanokey(n)] for n in vm.site_names()}
@torch.no_grad()
def evaluation():
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('exactly one assigned CUDA GPU required')
 torch.manual_seed(0);torch.cuda.reset_peak_memory_stats();device='cuda'
 conversion_record=json.loads((root/'CONVERSION.json').read_text());slim=root/'baseline.slim.pt'
 if sha(slim)!=conversion_record['slim_sha256']:raise RuntimeError('immutable converted checkpoint changed')
 raw=torch.load(slim,map_location='cpu',mmap=True,weights_only=False)
 if sha(vm.TARGET_DIR/'model_step_99999.safetensors')!=protocol['target_sha256'] or sha(vm.DATA)!=protocol['data_sha256']:raise RuntimeError('frozen target/data changed')
 target=vm.load_target(device);names=vm.site_names();cfg=nano.Config(**raw['config'])
 for n in names:
  if not torch.equal(target.site(n).W.cpu(),raw['native'][n]):raise RuntimeError('trained frozen native matrix differs '+n)
 dims={nanokey(n):raw['uv'][n]['V'].shape[0] for n in names};body=nano.CITransformer(dims,cfg.C_per_module,cfg);body.load_state_dict(raw['ci'],strict=True);body=body.to(device).eval()
 uv={n:(raw['uv'][n]['U'].to(device),raw['uv'][n]['V'].to(device)) for n in names};vpd=vm.VPD(target,Adapter(body),uv).eval()
 panel=json.loads((stage/'spec.json').read_text());rows=panel['rows'];episodes=panel['episodes'];passage_ids=sorted({e['passage'] for e in episodes}|{e['donor'] for e in episodes if 'donor' in e});export=Path.home()/protocol['evaluation_export_relative'];record=json.loads((export/'export.json').read_text());shape=record['files']['tokens']['shape'];token_file=export/'tokens.f64'
 if sha(token_file)!=protocol['evaluation_tokens_sha256'] or shape!=[2,513]:raise RuntimeError('fixed benchmark export token lineage changed')
 token_values=np.fromfile(token_file,dtype='<f8').reshape(shape)[:,:512]
 if not np.isfinite(token_values).all() or not np.equal(token_values,np.floor(token_values)).all():raise RuntimeError('invalid exported token IDs')
 ids=torch.from_numpy(token_values.astype(np.int64)).to(device)
 if rows!=512 or len(episodes)!=80 or passage_ids!=[0,1]:raise RuntimeError('frozen fullcontext two-passage eighty-episode scope required')
 all_on={n:torch.ones(1,rows,vpd.C[n],device=device) for n in names}
 edits={}
 for name,e in panel.get('edits',{}).items():
  shape=target.site(names[e['site']]).W.shape
  edits[name]=(torch.tensor(np.fromfile(stage/e['left'],dtype='<f8').reshape(shape[0],e['rank']),dtype=torch.float32,device=device),torch.tensor(np.fromfile(stage/e['right'],dtype='<f8').reshape(shape[1],e['rank']),dtype=torch.float32,device=device))
 def states(p,actions=(),donor=None,masks=None):
  vpd.clear();handles=[]
  for n in names:target.site(n).cache_input=target.site(n).cache_output=True
  by_site={}
  for a in actions:by_site.setdefault(a['site'],[]).append(a)
  for k,acts in by_site.items():
   st=target.site(names[k])
   def in_fn(x,acts=acts,k=k):
    x=x.clone()
    for a in acts:
     if a['type']=='scale_input':
      rs=slice(None) if a['row'] is None else slice(a['row'],a['row']+1);x[0,rs,a['cols'][0]:a['cols'][1]]*=a['scale']
     elif a['type']=='mix_input':
      r=a['row'];x[0,r]=(1-a['alpha'])*x[0,r]+a['alpha']*donor[(k,r,False)]
    return x
   def hook(module,inputs,output,acts=acts,k=k):
    for a in acts:
     if a['type']=='mix_output':
      r=a['row'];output[0,r]=(1-a['alpha'])*output[0,r]+a['alpha']*donor[(k,r,True)]
     elif a['type']=='add_map':
      left,right=edits[a['edit']];output=output+(module.last_input@right)@left.T
    module.last_output=output.detach();return output
   st.in_fn=in_fn;handles.append(st.register_forward_hook(hook))
  try:
   for n in names:target.site(n).mask=None if masks=='native' else (all_on[n] if masks is None else masks[n]);target.site(n).delta_mask=None
   logits=target(ids[p:p+1,:rows]);inputs={n:target.site(n).last_input for n in names};outputs={n:target.site(n).last_output for n in names}
  finally:
   for h in handles:h.remove()
   for n in names:target.site(n).in_fn=None
   vpd.clear()
  return logits,inputs,outputs
 def gates(inputs):return {n:(v>0).float() for n,v in vpd.ci_fn(inputs).items()}
 def changed(a,b):return sum(int((a[n]!=b[n]).sum().item()) for n in names)
 def fixed(p,actions=(),donor=None):
  _,x,_=states(p,actions,donor);on=gates(x);last=None
  for round_no in range(1,protocol['max_rounds']+1):
   _,x,_=states(p,actions,donor,on);again=gates(x);last=changed(on,again);on=again
   if last==0:return on,round_no,last
  return on,protocol['max_rounds'],last
 def kl(pred,truth,start=0):
  result=[]
  for at in range(start,rows,32):
   lp=F.log_softmax(truth[:,at:at+32].double(),-1);lq=F.log_softmax(pred[:,at:at+32].double(),-1);result.append((lp.exp()*(lp-lq)).sum(-1))
  values=torch.cat(result,dim=1).flatten();return {'mean':values.mean().item(),'worst_token':values.max().item(),'p50':values.quantile(.5).item(),'p95':values.quantile(.95).item(),'p99':values.quantile(.99).item(),'greedy_agreement':(pred[:,start:].argmax(-1)==truth[:,start:].argmax(-1)).float().mean().item()}
 # Adapter parity uses original pinned wrapper execution, no optimized fitting.
 small=ids[:1,:16];original=vm.load_target(device)
 for n in names:
  old=original.site(n);linear=torch.nn.Linear(old.W.shape[1],old.W.shape[0],bias=False,device=device);linear.weight.copy_(old.W);original.sites[n.replace('.','-')]=linear
 wrappers=nano.install_components(original,cfg.C_per_module);original.to(device)
 original.load_state_dict({**raw['fallback'],**{nanokey(n)+'.'+k:v for n,pair in raw['uv'].items() for k,v in pair.items()},**{nanokey(n)+'.W_target':v for n,v in raw['native'].items()}},strict=True)
 original_native=original(small);adapt_native=vpd.target_forward(small)
 for key,w in wrappers.items():w.mode='component';w.mask=torch.ones(1,16,w.C,device=device);w.delta_mask=torch.zeros(1,16,device=device)
 original_all=original(small);adapt_all=vpd.masked(small,{n:torch.ones(1,16,vpd.C[n],device=device) for n in names},None)
 parity={'native_logits_bitwise':torch.equal(original_native,adapt_native),'all_on_logits_bitwise':torch.equal(original_all,adapt_all),'native_max_abs':(original_native-adapt_native).abs().max().item(),'all_on_max_abs':(original_all-adapt_all).abs().max().item(),'component_tensors_exact':True,'CI_body':'same pinned upstream CITransformer; key adapter only','rows':16}
 if not parity['native_logits_bitwise'] or not parity['all_on_logits_bitwise']:save(root/'PARITY.json',parity);raise RuntimeError('original wrapper versus adapter bitwise parity failed')
 save(root/'PARITY.json',parity);del original,wrappers,original_native,adapt_native,original_all,adapt_all,raw;torch.cuda.empty_cache()
 # Same-parent Local is expressly native-fed and never supplies own-state gates.
 local_raw={n:[] for n in names}
 for p in passage_ids:
  _,x,y=states(p,masks='native');pre=vpd.ci_fn(x);on={n:(v>0).float() for n,v in pre.items()}
  for n in names:
   native_y=y[n].double();pred=((x[n]@uv[n][1])*on[n])@uv[n][0];numerator=(pred.double()-native_y).norm(dim=-1);native_norm=native_y.norm(dim=-1)
   local_raw[n].append((numerator,native_norm,on[n].sum(-1).mean().item()))
 local=[]
 for n,parts in local_raw.items():
  numerator=torch.cat([p[0].flatten() for p in parts]);native_norm=torch.cat([p[1].flatten() for p in parts]);denominator=native_norm.square().mean().sqrt()
  local.append({'site':n,'native_parent':True,'rows':1024,'RMS_native_output_row_L2':denominator.item(),'worst_relative_euclidean':(numerator/denominator).max().item() if denominator.item()>0 else None,'mean_active_components':sum(p[2] for p in parts)/len(parts)})
 save(root/'LOCAL_NATIVE_FED.json',{'scope':'same-parent sitewise diagnostics, native-fed CI; not autonomous evidence','rows':1024,'sites':local});del local_raw
 own_clean={};native_clean={}
 def donor(d,actions,feed):
  cache=native_clean if feed=='native' else own_clean
  if d not in cache:
   masks='native' if feed=='native' else fixed(d)[0];_,x,y=states(d,masks=masks);cache[d]=(x,y)
  x,y=cache[d];return {(a['site'],a['row'],a['type']=='mix_output'):(y if a['type']=='mix_output' else x)[names[a['site']]][0,a['row']] for a in actions if a['type'] in ('mix_input','mix_output')}
 records=[];journal=(root/'EPISODES.jsonl').open('x')
 for e in episodes:
  started=time.monotonic();nd=donor(e['donor'],e['actions'],'native') if 'donor' in e else None;truth,_,_=states(e['passage'],e['actions'],nd,'native');start=min((0 if a.get('row') is None else a['row']) for a in e['actions']) if e['actions'] else 0
  for feed in ('own','native'):
   d=donor(e['donor'],e['actions'],feed) if 'donor' in e else None
   if feed=='own':on,rounds,last=fixed(e['passage'],e['actions'],d)
   else:_,x,_=states(e['passage'],e['actions'],d,'native');on=gates(x);rounds,last=0,0
   pred,_,_=states(e['passage'],e['actions'],d,on);score=kl(pred,truth,start)
   r={'id':e['id'],'group':e['group'],'feed':feed,'passage':e['passage'],'rounds':rounds,'last_changed_coordinates':last,'fixedpoint_converged':last==0 if feed=='own' else None,'active_components_per_token':sum(v.sum().item() for v in on.values())/rows,'score_from':start,**score,'seconds_since_episode_start':time.monotonic()-started};journal.write(json.dumps(r)+'\n');journal.flush();records.append(r)
  print(f"episode {len(records)//2}/80 {e['id']} seconds={time.monotonic()-started:.3f}",flush=True)
 journal.close();groups={}
 for feed in ('own','native'):
  values={}
  for r in records:
   if r['feed']==feed:values.setdefault(r['group'],[]).append(r['mean'])
  groups[feed]={g:sum(v)/len(v) for g,v in values.items()}
 save(root/'REPORT.json',{'scope':protocol['scope'],'protocol':protocol,'conversion':conversion_record,'parity':parity,'group_mean_KL':groups,'worst_group_mean_KL':{f:max(g.values()) for f,g in groups.items()},'own_converged_episodes':sum(r['fixedpoint_converged'] is True for r in records if r['feed']=='own'),'own_total_episodes':80,'numeric_payload_bits_32':conversion_record['numeric_payload_bits_32'],'C32_total':None,'structural_cost_status':conversion_record['structural_cost_status'],'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'device':torch.cuda.get_device_name(),'evaluation_seconds':time.monotonic()-clock,'source_sha256':sha(Path(__file__)),'overall_acceptance_claim':False})
mode=sys.argv[1]
if mode=='convert':conversion()
elif mode=='evaluate':evaluation()
else:raise RuntimeError('explicit convert/evaluate mode required')

"""Bounded offline orchestration of pinned baseline conversion/evaluation only."""
import json,hashlib,time,subprocess,sys,os,resource
from pathlib import Path
stage=Path(__file__).resolve().parent;p=json.loads((stage/'PROTOCOL.json').read_text());mode=sys.argv[1];clock=time.monotonic()
if mode not in ('convert','evaluate'):raise RuntimeError('explicit conversion/evaluation mode required')
if hashlib.sha256((stage/'evaluate.py').read_bytes()).hexdigest()!=p['evaluation_source_sha256']:raise RuntimeError('frozen evaluator changed')
root=Path.home()/'mpd-data/cluster'/p['output'];root.mkdir(parents=True,exist_ok=True)
state={'mode':mode,'job_id':os.environ.get('SLURM_JOB_ID'),'state':'running','protocol_sha256':hashlib.sha256((stage/'PROTOCOL.json').read_bytes()).hexdigest(),'source_sha256':p['evaluation_source_sha256']}
try:
 with (root/(mode+'.stdout')).open('x') as stdout,(root/(mode+'.stderr')).open('x') as stderr:
  r=subprocess.run([sys.executable,str(stage/'evaluate.py'),mode],stdout=stdout,stderr=stderr,timeout=(840 if mode=='convert' else 1740)-(time.monotonic()-clock))
 state.update(returncode=r.returncode,state='complete' if r.returncode==0 else 'failed_or_unresolved')
 if r.returncode:raise RuntimeError('baseline diagnostic failed; no fidelity claim')
finally:
 state['seconds']=time.monotonic()-clock;state['peak_child_rss_bytes']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024;(root/(mode+'.STATE.json')).write_text(json.dumps(state,indent=2)+'\n')

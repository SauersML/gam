"""Compare completed query-law tests without accessing lexical heldout outputs."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parent);root=p.parse_args().data_root
modes=['zero','mean_native_delta','fitted']; colors=['#949bab','#bb91c4','#257d75']; labels=['Unchanged\nquery','Mean native\nshift','KL-fitted\nshift']
sources=[('QUERY_TAIL_DISCOVERY_V2','Largest query changes','Very little attention directly on color values'),('VALUE_QUERY_TAIL_DISCOVERY','Strongest value-token attention','77–79% of attention on color-value tokens')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12.5,6.8));fig.patch.set_facecolor('#faf9f6')
fig.suptitle('A good local fit is not yet a retrieval explanation',x=.065,ha='left',y=.96,fontsize=20,weight='bold')
fig.text(.065,.88,'Same rule: current query ≈ previous query + one shared shift per head',fontsize=12)
summary={}
for ax,(directory,title,subtitle) in zip(axes,sources):
 r=json.loads((root/directory/'REPORT.json').read_text());joint=[x for x in r['records'] if len(x['read_nodes'])==4];assert len(joint)==128
 means=[np.mean([x['metrics']['teacher_kl'] for x in joint if x['mode']==m]) for m in modes]
 maes=[np.mean([abs(x['metrics']['log_odds_error']) for x in joint if x['mode']==m]) for m in modes]
 ax.set_facecolor('#faf9f6');ax.bar(range(3),means,color=colors,width=.65)
 for i,v in enumerate(means):ax.text(i,v*1.22,f'{v:.3g}',ha='center',fontsize=12)
 ax.set(yscale='log',ylim=(3e-5,.5),xticks=range(3),xticklabels=labels,ylabel='Mean full-vocabulary output KL (nats)')
 ax.set_title(title+'\n'+subtitle,loc='left',pad=15,fontsize=12,weight='bold')
 ratio=means[2]/means[0]
 text=f'{1/ratio:.1f}× less output error' if ratio<1 else f'{ratio:.2f}× more output error'
 ax.text(.5,.97,text,transform=ax.transAxes,ha='center',va='top',fontsize=14,weight='bold',color='#257d75' if ratio<1 else '#b34b37')
 summary[directory]={'mean_kl':dict(zip(modes,map(float,means))),'answer_log_odds_mae':dict(zip(modes,map(float,maes))),'fit_over_zero_kl':float(ratio)}
fig.text(.065,.135,'Each panel changes four selected heads together. 32 discovery pairs; shifts fitted on the opposite fold.',fontsize=10)
fig.text(.065,.091,'All keys, values and remaining native computation stay live. Head selection used discovery measurements.',fontsize=10,color='#555555')
fig.text(.065,.052,'The additive rule was supplied; all fits reached the iteration limit. Failure does not yet separate fitting from missing state.',fontsize=9,color='#555555')
fig.text(.065,.017,'No lexical heldout results or native parameter-edit results are included. These tests do not establish a retrieval algorithm.',fontsize=9,color='#555555')
fig.subplots_adjust(left=.09,right=.97,top=.73,bottom=.26,wspace=.27)
fig.savefig(root/'query_rule_scope.pdf',facecolor=fig.get_facecolor());fig.savefig(root/'query_rule_scope.png',dpi=170,facecolor=fig.get_facecolor())
(root/'QUERY_SCOPE_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

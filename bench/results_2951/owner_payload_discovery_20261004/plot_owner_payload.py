"""Aggregate completed native interventions and plot fixed physical owner comparisons."""
import argparse, json, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parent)
r=p.parse_args().data_root
report=json.loads((r/'OWNER_PAYLOAD_DISCOVERY/REPORT.json').read_text())
cases=report['cases'];assert len(cases)==64
key='fixed_color_log_odds_central_response_per_epsilon_fraction'
groups={}
for c in cases:
 f=c['factors'].copy();owner=f.pop('query_owner')
 groups.setdefault(json.dumps(f,sort_keys=True),{})[owner]=c
assert len(groups)==32 and all(len(x)==2 for x in groups.values())
pairs=[(g[0],g[1]) for g in groups.values()]
summary={'source_commit':'6a5a635c27','seconds':report['seconds'],'cases':64,'signed_controls':256,'query_swap_pairs':32,'by_query_now':{},'scope':'Native Qwen3-0.6B L19H5 (zero-based). Supplied record spans, selected head and color contrast; measured native causal transport, not recovered address construction. All discovery; no heldout.', 'intervention':'Both signs of0.1*d applied to fixed owner old/new value slots, only this head last-query invocation; shared weights unchanged, native suffix live.'}
for now in [0,1]:
 rows=[c for c in cases if c['factors']['query_now']==now]
 own=[c['owner_directional_responses'][c['factors']['query_owner']][key]*.1 for c in rows]
 other=[c['owner_directional_responses'][1-c['factors']['query_owner']][key]*.1 for c in rows]
 summary['by_query_now'][str(now)]={'cases':len(rows),'mean_symmetric_odds_response_queried_owner':statistics.mean(own),'mean_symmetric_odds_response_other_owner':statistics.mean(other),'queried_positive':sum(x>0 for x in own),'queried_stronger':sum(abs(x)>abs(y) for x,y in zip(own,other))}
summary['fixed_owner0_attention_larger_when_queried']=sum(a['owner_record_attention_masses'][0]>b['owner_record_attention_masses'][0] for a,b in pairs)
summary['fixed_owner0_payload_effect_larger_when_queried']=sum(a['owner_directional_responses'][0][key]>b['owner_directional_responses'][0][key] for a,b in pairs)
summary['max_transport_identity_error']=max(c['direct_raw_attend_transport_check']['maximum_absolute_error'] for x in cases for c in x['controls'])
summary['max_earlier_position_change']=max(c['maximum_other_query_position_change'] for x in cases for c in x['controls'])
summary['max_prefix_direction_invariance_error']=max(p[k]['maximum_absolute_error'] for name in ['query_owner_prefix_direction_invariance','query_now_prefix_direction_invariance'] for p in report[name] for k in ['prefix_key_check','prefix_value_check','direction_check'])
(r/'OWNER_PAYLOAD_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(12,7));fig.patch.set_facecolor('#faf9f6')
fig.suptitle('One head routes color information by the queried person',x=.07,ha='left',y=.96,fontsize=19,weight='bold')
fig.text(.07,.88,'Change only the person in the question; keep the records, colors and perturbation fixed.',fontsize=12)
colors={0:'#8c9cab',1:'#247d75'}
for a,b in pairs:
 now=a['factors']['query_now'];color=colors[now]
 ys=[[a['owner_record_attention_masses'][0],b['owner_record_attention_masses'][0]],[a['owner_directional_responses'][0][key]*.1,b['owner_directional_responses'][0][key]*.1]]
 for ax,y in zip(axs,ys):ax.plot([0,1],y,'o-',color=color,alpha=.45,linewidth=1,markersize=4)
for ax in axs:
 ax.set_facecolor('#faf9f6');ax.set_xticks([0,1],['Question asks\nabout Alice','Question asks\nabout Bob']);ax.set_xlim(-.18,1.18);ax.grid(axis='y',alpha=.15)
axs[0].set_title('Attention to Alice’s two color records',loc='left',weight='bold',fontsize=12,pad=15);axs[0].set_ylabel('Fraction of this head’s attention');axs[0].set_ylim(0,1)
axs[1].set_title('Effect of changing Alice’s stored color payloads',loc='left',weight='bold',fontsize=12,pad=15);axs[1].set_ylabel('Symmetric change in color log odds (nats)');axs[1].axhline(0,color='#777',linewidth=.7)
for ax in axs:
 for now,label in [(0,'Without “now”'),(1,'With “now”')]:ax.plot([],[],color=colors[now],marker='o',label=label)
axs[0].legend(frameon=False,loc='upper right',fontsize=10)
fig.text(.07,.175,'32 matched query swaps; all show stronger Alice-record attention and payload effect when Alice is queried.',fontsize=10)
fig.text(.07,.123,'Payload test: ±0.1 × native color contrast, applied to Alice’s old and new record values at one head invocation.',fontsize=10)
fig.text(.07,.081,'Effect = [color log odds(+perturbation) − color log odds(−perturbation)] / 2. Full native downstream runs.',fontsize=10)
fig.text(.07,.038,'Qwen3-0.6B · layer 19, head 5 (zero-based) · 64 discovery cases · supplied spans · address construction not yet explained.',fontsize=9,color='#555')
fig.subplots_adjust(left=.09,right=.97,top=.75,bottom=.31,wspace=.3)
fig.savefig(r/'owner_payload.pdf',facecolor=fig.get_facecolor());fig.savefig(r/'owner_payload.png',dpi=170,facecolor=fig.get_facecolor())
print(json.dumps(summary,indent=2))

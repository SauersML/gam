import json,pathlib,hashlib,itertools
from tokenizers import Tokenizer
root=pathlib.Path('/Users/user/mpd-data/codex/qwen-association-discrimination-20261004');root.mkdir(exist_ok=True)
tp=pathlib.Path('/Users/user/mpd-data/codex/native-capability-screen-qwen06-20261004/tokenizer.json');t=Tokenizer.from_file(str(tp))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def span_records(text,offsets,parts):
 out=[]
 for role,a,b in parts:
  out.append({'role':role,'char_start':a,'char_end':b,'text':text[a:b],'token_indices':[i for i,(s,e) in enumerate(offsets) if s<b and e>a]})
 return out
panels={}
for panel,sets in [('discovery',[(('Alice','Bob'),('blue','green')),(('John','Mary'),('blue','green'))]),('heldout',[(('Carol','David'),('red','yellow')),(('Emma','Frank'),('orange','purple'))])]:
 cases=[]
 for set_id,query,order,update_order,swap,update_style,query_style in itertools.product(range(1),range(2),range(2),range(2),range(2),range(2),range(2)):
  owners,colors=sets[set_id];old=colors[::-1] if swap else colors;new=old[::-1];parts=[];text=''
  def append(statement,roles):
   # Locate spans in this statement before concatenation; no model-derived annotations.
   base=len(text_holder[0]);text_holder[0]+=statement
   for role,word in roles:
    at=statement.index(word);parts.append((role,base+at,base+at+len(word)))
  text_holder=['']
  for i in ([0,1] if order==0 else [1,0]):
   statement=f" {owners[i]}'s color is {old[i]}."
   append(statement,[(f'old_owner_{i}',owners[i]),(f'old_value_{i}',old[i])])
  # Independently cross old and update order to isolate recency hypotheses.
  for i in ([0,1] if update_order==0 else [1,0]):
   if panel=='discovery':statement=f" {owners[i]}'s color is now {new[i]}." if update_style==0 else f" {owners[i]}'s color has changed to {new[i]}."
   else:statement=f" {owners[i]} now has the color {new[i]}." if update_style==0 else f" The updated color for {owners[i]} is {new[i]}."
   append(statement,[(f'update_owner_{i}',owners[i]),(f'new_value_{i}',new[i])])
  if panel=='discovery':cue=f" {owners[query]}'s color is"+(' now' if query_style else '')
  else:cue=f" The color of {owners[query]} is"+(' now' if query_style else '')
  append(cue,[('query_owner',owners[query]),('query_cue',cue.strip())]);text=text_holder[0];e=t.encode(text,add_special_tokens=False)
  targets={}
  for role,value in [('updated_owner_value',new[query]),('old_owner_value',old[query])]:
   token=t.encode(' '+value,add_special_tokens=False);assert len(token.ids)==1,(value,token.ids);targets[role]={'text':' '+value,'id':token.ids[0],'piece':token.tokens[0]}
  cases.append({'id':f'{panel}-{len(cases):03d}','panel':panel,'factors':{'lexical_set':set_id,'query_owner':query,'old_statement_order':order,'update_statement_order':update_order,'payload_swap':swap,'update_wording':update_style,'query_now':query_style},'prompt':text,'tokens':e.ids,'offsets':e.offsets,'spans':span_records(text,e.offsets,parts),'hypothesis_targets':targets,'target_ontology_scope':'Updated-owner vs old-owner are rival human-specified interpretations, not observed model labels or unsupervised role discovery.'})
 assert len(cases)==64;assert len(set(c['prompt'] for c in cases))==64
 panels[panel]={'version':1,'panel':panel,'tokenizer_sha256':sha(tp),'cases':cases};(root/f'{panel.upper()}.json').write_text(json.dumps(panels[panel],indent=2)+'\n')
protocol={'version':1,'status':'Frozen before any new neural evaluation; heldout outputs not inspected.','model':'Qwen3-0.6B full28L','checkpoint_sha256':'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b','tokenizer_source':'mats:/mnt/nw/home/s.sauers/mpd-data/cluster/qwen3_0p6b_data/model/tokenizer.json','tokenizer_sha256':sha(tp),'development_data':'The earlier42-prompt Qwen capability screen is development data and is not heldout evidence.','panels':{p:{'file':f'{p.upper()}.json','sha256':sha(root/f'{p.upper()}.json'),'cases':64,'max_tokens':max(len(c['tokens']) for c in panels[p]['cases'])} for p in panels},'rival_hypotheses':['Owner-specific updated association','Retrieval of old association from repeated query wording','Most recently mentioned value regardless of owner','Lexical continuation/value prior'],'scope':'Human ontology/labels propose rival hypotheses. Native role/law inference remains a fitting/search task. No unsupervised-role-discovery or algorithm-recovery claim from this deck.','selection_rule':'Use discovery only for fitting, structural selection and stopping. Freeze executable candidate IDs and settings before any heldout model outputs; evaluate all64 heldout cases once.','metrics_plan':['Native updated-vs-old log odds and absolute probabilities','Contrasts by queried owner, payload swap, fact/update order, query now, update wording','Saved explanation same-state/native-intervention prediction errors after independently frozen search'],'limitations':['Only one entity pair/value pair per panel; lexical generalization is restricted to the declared heldout pair.','All updated values are swapped between two owners; generalization beyond this declared family untested.','Native capability is not mechanism recovery; directions and role annotations do not supply an executable learned law.'],'no_execution':True}
(root/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n');print(root);print(protocol['panels'])

import sys, re, collections
f=sys.argv[1]; top=int(sys.argv[2]) if len(sys.argv)>2 else 40
filt=sys.argv[3] if len(sys.argv)>3 else None
incl=collections.Counter(); selfc=collections.Counter(); tot=0
def clean(fr):
    fr=re.sub(r'\s*\(.*?\)\s*$','',fr)
    fr=re.sub(r'::h[0-9a-f]{16}$','',fr)
    return fr[:160]
for line in open(f):
    line=line.rstrip('\n')
    if not line: continue
    st,c=line.rsplit(' ',1); c=int(c)
    frames=[clean(x) for x in st.split(';')]
    if filt and not any(filt in x for x in frames): continue
    tot+=c
    for fr in set(frames): incl[fr]+=c
    selfc[frames[-1]]+=c
print("total samples",tot)
print("--- inclusive")
for k,v in incl.most_common(top): print(f"{100*v/tot:5.1f}% {k}")
print("--- self")
for k,v in selfc.most_common(25): print(f"{100*v/tot:5.1f}% {k}")

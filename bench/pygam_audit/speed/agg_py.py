"""Collapse py-spy --native raw stacks to their Python-only frames (native hex dropped).
Usage: agg_py.py FILE [depth]"""
import sys, re, collections
f = sys.argv[1]; depth = int(sys.argv[2]) if len(sys.argv) > 2 else 6
c = collections.Counter(); tot = 0
for line in open(f):
    line = line.rstrip("\n")
    if not line: continue
    st, n = line.rsplit(" ", 1); n = int(n); tot += n
    fr = [x for x in st.split(";") if not re.match(r"^0x[0-9a-f]+", x.strip())]
    fr = [re.sub(r"\s*\((.*?)\)", lambda m: "(" + m.group(1).split("/")[-1] + ")", x) for x in fr]
    c[";".join(fr[:depth])] += n
print("total", tot)
for k, v in c.most_common(30): print(f"{100*v/tot:5.1f}% {k}")

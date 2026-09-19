"""Streaming variant of mc_sample.py: one JSON line per replicate. Usage: python mc_sample_stream.py <cell> <first> <last>"""
import sys, json
from mc_sample import one_rep
cell, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for k in range(a, b):
    print(json.dumps(one_rep((cell, k))), flush=True)

"""Behavior code_variable.loop (vpd4l; M's top token is right on 100% of the targets): Python variable
reuse: in a for loop body, the loop variable is printed.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 9.74 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 9.73, L3.H4 7.75, L2.H3 7.23, layer 3 MLP 5.95, L3.H5 3.06, layer 1 MLP
1.30 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 9.73 bits
mlp1 = node(L[1].mlp)  # recovers 1.30 bits
h2_3 = node(L[2].head[3])  # recovers 7.23 bits
h3_4 = node(L[3].head[4])  # recovers 7.75 bits
h3_5 = node(L[3].head[5])  # recovers 3.06 bits
mlp3 = node(L[3].mlp)  # recovers 5.95 bits

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_3,
    embed >> h3_4,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_3,
    mlp0 >> h3_4,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    mlp1 >> h2_3,
    mlp1 >> h3_4,
    mlp1 >> h3_5,
    mlp1 >> mlp3,
    h2_3 >> h3_4,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    h3_4 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_3 >> logits,
    h3_4 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)


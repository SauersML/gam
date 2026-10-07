"""Behavior list_copy.first (vpd4l; M's top token is right on 27% of the targets): List indexing: given
a list of four words, the first word; the counterfactual swaps the first word with another, so the
same words appear.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 1.56 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 1.55, L3.H4 1.14, L2.H3 0.93, layer 3 MLP 0.81, L3.H5 0.64, L2.H4 0.21
bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 1.55 bits
h2_3 = node(L[2].head[3])  # recovers 0.93 bits
h2_4 = node(L[2].head[4])  # recovers 0.21 bits
h3_4 = node(L[3].head[4])  # recovers 1.14 bits
h3_5 = node(L[3].head[5])  # recovers 0.64 bits
mlp3 = node(L[3].mlp)  # recovers 0.81 bits

edges(
    embed >> mlp0,
    embed >> h2_3,
    embed >> h2_4,
    embed >> h3_4,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> h3_4,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    h2_3 >> h3_4,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    h2_4 >> h3_4,
    h2_4 >> h3_5,
    h2_4 >> mlp3,
    h3_4 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    h3_4 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)


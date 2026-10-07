"""Behavior html_close.nested (vpd4l; M's top token is right on 91% of the targets): HTML tag closing:
after an inner element is closed, the next closing tag names the outer element.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 10.58 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 10.55, L2.H3 9.44, layer 3 MLP 6.41, L3.H4 3.39, L3.H5 2.29, L2.H2 2.03,
L2.H4 1.99, layer 1 MLP 1.32 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 10.55 bits
mlp1 = node(L[1].mlp)  # recovers 1.32 bits
h2_2 = node(L[2].head[2])  # recovers 2.03 bits
h2_3 = node(L[2].head[3])  # recovers 9.44 bits
h2_4 = node(L[2].head[4])  # recovers 1.99 bits
h3_4 = node(L[3].head[4])  # recovers 3.39 bits
h3_5 = node(L[3].head[5])  # recovers 2.29 bits
mlp3 = node(L[3].mlp)  # recovers 6.41 bits

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_2,
    embed >> h2_3,
    embed >> h2_4,
    embed >> h3_4,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_2,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> h3_4,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    mlp1 >> h2_2,
    mlp1 >> h2_3,
    mlp1 >> h2_4,
    mlp1 >> h3_4,
    mlp1 >> h3_5,
    mlp1 >> mlp3,
    h2_2 >> h3_4,
    h2_2 >> h3_5,
    h2_2 >> mlp3,
    h2_3 >> h3_4,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    h2_4 >> h3_4,
    h2_4 >> h3_5,
    h2_4 >> mlp3,
    h3_4 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_2 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    h3_4 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)


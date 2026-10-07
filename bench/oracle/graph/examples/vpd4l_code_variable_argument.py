"""Behavior code_variable.argument (vpd4l; M's top token is right on 100% of the targets): Python
variable reuse: a function body uses its only argument.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 14.45 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 14.44, L3.H4 11.94, L2.H3 11.79, layer 3 MLP 9.98, L3.H5 4.43, L2.H2
2.50, L2.H4 1.99 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 14.44 bits
h2_2 = node(L[2].head[2])  # recovers 2.50 bits
h2_3 = node(L[2].head[3])  # recovers 11.79 bits
h2_4 = node(L[2].head[4])  # recovers 1.99 bits
h3_4 = node(L[3].head[4])  # recovers 11.94 bits
h3_5 = node(L[3].head[5])  # recovers 4.43 bits
mlp3 = node(L[3].mlp)  # recovers 9.98 bits

edges(
    embed >> mlp0,
    embed >> h2_2,
    embed >> h2_3,
    embed >> h2_4,
    embed >> h3_4,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> h2_2,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> h3_4,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
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
    h2_2 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    h3_4 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)


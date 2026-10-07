"""Behavior quote_type.python_string (vpd4l; M's top token is right on 46% of the targets): Quote
matching: a Python string opened with a single or a double quote is closed with the same mark; the
counterfactual opens with the other mark.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 5.99 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 5.99, L2.H1 5.95, layer 1 MLP 2.67, layer 3 MLP 1.69, L3.H3 1.41, L2.H2
0.86, L3.H1 0.80, L3.H0 0.60 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 5.99 bits
mlp1 = node(L[1].mlp)  # recovers 2.67 bits
h2_1 = node(L[2].head[1])  # recovers 5.95 bits
h2_2 = node(L[2].head[2])  # recovers 0.86 bits
h3_0 = node(L[3].head[0])  # recovers 0.60 bits
h3_1 = node(L[3].head[1])  # recovers 0.80 bits
h3_3 = node(L[3].head[3])  # recovers 1.41 bits
mlp3 = node(L[3].mlp)  # recovers 1.69 bits

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_1,
    embed >> h2_2,
    embed >> h3_0,
    embed >> h3_1,
    embed >> h3_3,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_1,
    mlp0 >> h2_2,
    mlp0 >> h3_0,
    mlp0 >> h3_1,
    mlp0 >> h3_3,
    mlp0 >> mlp3,
    mlp1 >> h2_1,
    mlp1 >> h2_2,
    mlp1 >> h3_0,
    mlp1 >> h3_1,
    mlp1 >> h3_3,
    mlp1 >> mlp3,
    h2_1 >> h3_0,
    h2_1 >> h3_1,
    h2_1 >> h3_3,
    h2_1 >> mlp3,
    h2_2 >> h3_0,
    h2_2 >> h3_1,
    h2_2 >> h3_3,
    h2_2 >> mlp3,
    h3_0 >> mlp3,
    h3_1 >> mlp3,
    h3_3 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_1 >> logits,
    h2_2 >> logits,
    h3_0 >> logits,
    h3_1 >> logits,
    h3_3 >> logits,
    mlp3 >> logits,
)


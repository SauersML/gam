"""Behavior quote_close.said (vpd4l; M's top token is right on 9% of the targets): Quote closing: after
a quoted sentence ends with a period, the next token closes the quotation; the counterfactual opens
a parenthesis instead.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 3.40 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 3.39, L2.H1 3.09, L3.H3 2.12, layer 1 MLP 2.08, layer 2 MLP 1.29, layer 3
MLP 1.06, L0.H1 0.82, L1.H4 0.76, L3.H1 0.62, L1.H5 0.49 bits. The program lets every write among
them reach every later read.
"""
from mech import node, edges, L, embed, logits

h0_1 = node(L[0].head[1])  # recovers 0.82 bits
mlp0 = node(L[0].mlp)  # recovers 3.39 bits
h1_4 = node(L[1].head[4])  # recovers 0.76 bits
h1_5 = node(L[1].head[5])  # recovers 0.49 bits
mlp1 = node(L[1].mlp)  # recovers 2.08 bits
h2_1 = node(L[2].head[1])  # recovers 3.09 bits
mlp2 = node(L[2].mlp)  # recovers 1.29 bits
h3_1 = node(L[3].head[1])  # recovers 0.62 bits
h3_3 = node(L[3].head[3])  # recovers 2.12 bits
mlp3 = node(L[3].mlp)  # recovers 1.06 bits

edges(
    embed >> h0_1,
    embed >> mlp0,
    embed >> h1_4,
    embed >> h1_5,
    embed >> mlp1,
    embed >> h2_1,
    embed >> mlp2,
    embed >> h3_1,
    embed >> h3_3,
    embed >> mlp3,
    h0_1 >> mlp0,
    h0_1 >> h1_4,
    h0_1 >> h1_5,
    h0_1 >> mlp1,
    h0_1 >> h2_1,
    h0_1 >> mlp2,
    h0_1 >> h3_1,
    h0_1 >> h3_3,
    h0_1 >> mlp3,
    mlp0 >> h1_4,
    mlp0 >> h1_5,
    mlp0 >> mlp1,
    mlp0 >> h2_1,
    mlp0 >> mlp2,
    mlp0 >> h3_1,
    mlp0 >> h3_3,
    mlp0 >> mlp3,
    h1_4 >> mlp1,
    h1_4 >> h2_1,
    h1_4 >> mlp2,
    h1_4 >> h3_1,
    h1_4 >> h3_3,
    h1_4 >> mlp3,
    h1_5 >> mlp1,
    h1_5 >> h2_1,
    h1_5 >> mlp2,
    h1_5 >> h3_1,
    h1_5 >> h3_3,
    h1_5 >> mlp3,
    mlp1 >> h2_1,
    mlp1 >> mlp2,
    mlp1 >> h3_1,
    mlp1 >> h3_3,
    mlp1 >> mlp3,
    h2_1 >> mlp2,
    h2_1 >> h3_1,
    h2_1 >> h3_3,
    h2_1 >> mlp3,
    mlp2 >> h3_1,
    mlp2 >> h3_3,
    mlp2 >> mlp3,
    h3_1 >> mlp3,
    h3_3 >> mlp3,
    h0_1 >> logits,
    mlp0 >> logits,
    h1_4 >> logits,
    h1_5 >> logits,
    mlp1 >> logits,
    h2_1 >> logits,
    mlp2 >> logits,
    h3_1 >> logits,
    h3_3 >> logits,
    mlp3 >> logits,
)


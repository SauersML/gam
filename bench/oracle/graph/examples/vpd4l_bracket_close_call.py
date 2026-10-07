"""Behavior bracket_close.call (vpd4l; M's top token is right on 50% of the targets): Bracket closing
in function calls: after the last argument the next token closes the open calls.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 3.58 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 3.54, L2.H1 2.61, L0.H1 1.75, L1.H0 1.59, L1.H4 1.34, layer 2 MLP 1.18,
L3.H3 1.17, L1.H1 1.03, L3.H1 0.81, L2.H0 0.55, L1.H3 0.50 bits. The program lets every write among
them reach every later read.
"""
from mech import node, edges, L, embed, logits

h0_1 = node(L[0].head[1])  # recovers 1.75 bits
mlp0 = node(L[0].mlp)  # recovers 3.54 bits
h1_0 = node(L[1].head[0])  # recovers 1.59 bits
h1_1 = node(L[1].head[1])  # recovers 1.03 bits
h1_3 = node(L[1].head[3])  # recovers 0.50 bits
h1_4 = node(L[1].head[4])  # recovers 1.34 bits
h2_0 = node(L[2].head[0])  # recovers 0.55 bits
h2_1 = node(L[2].head[1])  # recovers 2.61 bits
mlp2 = node(L[2].mlp)  # recovers 1.18 bits
h3_1 = node(L[3].head[1])  # recovers 0.81 bits
h3_3 = node(L[3].head[3])  # recovers 1.17 bits

edges(
    embed >> h0_1,
    embed >> mlp0,
    embed >> h1_0,
    embed >> h1_1,
    embed >> h1_3,
    embed >> h1_4,
    embed >> h2_0,
    embed >> h2_1,
    embed >> mlp2,
    embed >> h3_1,
    embed >> h3_3,
    h0_1 >> mlp0,
    h0_1 >> h1_0,
    h0_1 >> h1_1,
    h0_1 >> h1_3,
    h0_1 >> h1_4,
    h0_1 >> h2_0,
    h0_1 >> h2_1,
    h0_1 >> mlp2,
    h0_1 >> h3_1,
    h0_1 >> h3_3,
    mlp0 >> h1_0,
    mlp0 >> h1_1,
    mlp0 >> h1_3,
    mlp0 >> h1_4,
    mlp0 >> h2_0,
    mlp0 >> h2_1,
    mlp0 >> mlp2,
    mlp0 >> h3_1,
    mlp0 >> h3_3,
    h1_0 >> h2_0,
    h1_0 >> h2_1,
    h1_0 >> mlp2,
    h1_0 >> h3_1,
    h1_0 >> h3_3,
    h1_1 >> h2_0,
    h1_1 >> h2_1,
    h1_1 >> mlp2,
    h1_1 >> h3_1,
    h1_1 >> h3_3,
    h1_3 >> h2_0,
    h1_3 >> h2_1,
    h1_3 >> mlp2,
    h1_3 >> h3_1,
    h1_3 >> h3_3,
    h1_4 >> h2_0,
    h1_4 >> h2_1,
    h1_4 >> mlp2,
    h1_4 >> h3_1,
    h1_4 >> h3_3,
    h2_0 >> mlp2,
    h2_0 >> h3_1,
    h2_0 >> h3_3,
    h2_1 >> mlp2,
    h2_1 >> h3_1,
    h2_1 >> h3_3,
    mlp2 >> h3_1,
    mlp2 >> h3_3,
    h0_1 >> logits,
    mlp0 >> logits,
    h1_0 >> logits,
    h1_1 >> logits,
    h1_3 >> logits,
    h1_4 >> logits,
    h2_0 >> logits,
    h2_1 >> logits,
    mlp2 >> logits,
    h3_1 >> logits,
    h3_3 >> logits,
)


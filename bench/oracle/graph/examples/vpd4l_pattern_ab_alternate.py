"""Behavior pattern_ab.alternate (vpd4l; M's top token is right on 88% of the targets): Alternation:
two words alternate several times; the next word continues the alternation; the counterfactual
starts the alternation with the other word.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 3.46 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 3.36, L1.H1 2.96, L2.H4 2.93, layer 3 MLP 2.38, L2.H3 1.40, layer 1 MLP
1.25, L3.H5 1.10, L0.H4 0.85 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

h0_4 = node(L[0].head[4])  # recovers 0.85 bits
mlp0 = node(L[0].mlp)  # recovers 3.36 bits
h1_1 = node(L[1].head[1])  # recovers 2.96 bits
mlp1 = node(L[1].mlp)  # recovers 1.25 bits
h2_3 = node(L[2].head[3])  # recovers 1.40 bits
h2_4 = node(L[2].head[4])  # recovers 2.93 bits
h3_5 = node(L[3].head[5])  # recovers 1.10 bits
mlp3 = node(L[3].mlp)  # recovers 2.38 bits

edges(
    embed >> h0_4,
    embed >> mlp0,
    embed >> h1_1,
    embed >> mlp1,
    embed >> h2_3,
    embed >> h2_4,
    embed >> h3_5,
    embed >> mlp3,
    h0_4 >> mlp0,
    h0_4 >> h1_1,
    h0_4 >> mlp1,
    h0_4 >> h2_3,
    h0_4 >> h2_4,
    h0_4 >> h3_5,
    h0_4 >> mlp3,
    mlp0 >> h1_1,
    mlp0 >> mlp1,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    h1_1 >> mlp1,
    h1_1 >> h2_3,
    h1_1 >> h2_4,
    h1_1 >> h3_5,
    h1_1 >> mlp3,
    mlp1 >> h2_3,
    mlp1 >> h2_4,
    mlp1 >> h3_5,
    mlp1 >> mlp3,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    h2_4 >> h3_5,
    h2_4 >> mlp3,
    h3_5 >> mlp3,
    h0_4 >> logits,
    mlp0 >> logits,
    h1_1 >> logits,
    mlp1 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)


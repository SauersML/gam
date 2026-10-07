"""Behavior alphabet.lower (vpd4l; M's top token is right on 61% of the targets): Alphabet succession:
a run of lowercase letters; the next letter continues it.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 1.64 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 1.63, L3.H0 1.08, layer 3 MLP 0.89, layer 2 MLP 0.59, layer 1 MLP 0.50,
L2.H2 0.39, L2.H4 0.32, L2.H3 0.27, L1.H1 0.21, L0.H0 0.18 bits. The program lets every write among
them reach every later read.
"""
from mech import node, edges, L, embed, logits

h0_0 = node(L[0].head[0])  # recovers 0.18 bits
mlp0 = node(L[0].mlp)  # recovers 1.63 bits
h1_1 = node(L[1].head[1])  # recovers 0.21 bits
mlp1 = node(L[1].mlp)  # recovers 0.50 bits
h2_2 = node(L[2].head[2])  # recovers 0.39 bits
h2_3 = node(L[2].head[3])  # recovers 0.27 bits
h2_4 = node(L[2].head[4])  # recovers 0.32 bits
mlp2 = node(L[2].mlp)  # recovers 0.59 bits
h3_0 = node(L[3].head[0])  # recovers 1.08 bits
mlp3 = node(L[3].mlp)  # recovers 0.89 bits

edges(
    embed >> h0_0,
    embed >> mlp0,
    embed >> h1_1,
    embed >> mlp1,
    embed >> h2_2,
    embed >> h2_3,
    embed >> h2_4,
    embed >> mlp2,
    embed >> h3_0,
    embed >> mlp3,
    h0_0 >> mlp0,
    h0_0 >> h1_1,
    h0_0 >> mlp1,
    h0_0 >> h2_2,
    h0_0 >> h2_3,
    h0_0 >> h2_4,
    h0_0 >> mlp2,
    h0_0 >> h3_0,
    h0_0 >> mlp3,
    mlp0 >> h1_1,
    mlp0 >> mlp1,
    mlp0 >> h2_2,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> mlp2,
    mlp0 >> h3_0,
    mlp0 >> mlp3,
    h1_1 >> mlp1,
    h1_1 >> h2_2,
    h1_1 >> h2_3,
    h1_1 >> h2_4,
    h1_1 >> mlp2,
    h1_1 >> h3_0,
    h1_1 >> mlp3,
    mlp1 >> h2_2,
    mlp1 >> h2_3,
    mlp1 >> h2_4,
    mlp1 >> mlp2,
    mlp1 >> h3_0,
    mlp1 >> mlp3,
    h2_2 >> mlp2,
    h2_2 >> h3_0,
    h2_2 >> mlp3,
    h2_3 >> mlp2,
    h2_3 >> h3_0,
    h2_3 >> mlp3,
    h2_4 >> mlp2,
    h2_4 >> h3_0,
    h2_4 >> mlp3,
    mlp2 >> h3_0,
    mlp2 >> mlp3,
    h3_0 >> mlp3,
    h0_0 >> logits,
    mlp0 >> logits,
    h1_1 >> logits,
    mlp1 >> logits,
    h2_2 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    mlp2 >> logits,
    h3_0 >> logits,
    mlp3 >> logits,
)


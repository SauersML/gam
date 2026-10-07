"""Behavior pronoun_gender.said (vpd4l; M's top token is right on 98% of the targets): Gendered
pronoun: the next word is the pronoun for the named person; the counterfactual swaps the name's
gender.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 1.00 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 0.99, L3.H4 0.83, layer 2 MLP 0.45, layer 3 MLP 0.32, L3.H2 0.26, layer 1
MLP 0.22, L0.H1 0.12, L2.H2 0.10 bits. The program lets every write among them reach every later
read.
"""
from mech import node, edges, L, embed, logits

h0_1 = node(L[0].head[1])  # recovers 0.12 bits
mlp0 = node(L[0].mlp)  # recovers 0.99 bits
mlp1 = node(L[1].mlp)  # recovers 0.22 bits
h2_2 = node(L[2].head[2])  # recovers 0.10 bits
mlp2 = node(L[2].mlp)  # recovers 0.45 bits
h3_2 = node(L[3].head[2])  # recovers 0.26 bits
h3_4 = node(L[3].head[4])  # recovers 0.83 bits
mlp3 = node(L[3].mlp)  # recovers 0.32 bits

edges(
    embed >> h0_1,
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_2,
    embed >> mlp2,
    embed >> h3_2,
    embed >> h3_4,
    embed >> mlp3,
    h0_1 >> mlp0,
    h0_1 >> mlp1,
    h0_1 >> h2_2,
    h0_1 >> mlp2,
    h0_1 >> h3_2,
    h0_1 >> h3_4,
    h0_1 >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_2,
    mlp0 >> mlp2,
    mlp0 >> h3_2,
    mlp0 >> h3_4,
    mlp0 >> mlp3,
    mlp1 >> h2_2,
    mlp1 >> mlp2,
    mlp1 >> h3_2,
    mlp1 >> h3_4,
    mlp1 >> mlp3,
    h2_2 >> mlp2,
    h2_2 >> h3_2,
    h2_2 >> h3_4,
    h2_2 >> mlp3,
    mlp2 >> h3_2,
    mlp2 >> h3_4,
    mlp2 >> mlp3,
    h3_2 >> mlp3,
    h3_4 >> mlp3,
    h0_1 >> logits,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_2 >> logits,
    mlp2 >> logits,
    h3_2 >> logits,
    h3_4 >> logits,
    mlp3 >> logits,
)


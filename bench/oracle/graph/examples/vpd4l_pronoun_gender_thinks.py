"""Behavior pronoun_gender.thinks (vpd4l; M's top token is right on 0% of the targets): Gendered
pronoun: the next word is the pronoun for the named person; the counterfactual swaps the name's
gender.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 1.09 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 1.09, L3.H4 1.00, layer 2 MLP 0.57, L3.H2 0.39, layer 3 MLP 0.35, layer 1
MLP 0.20, L0.H1 0.12 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

h0_1 = node(L[0].head[1])  # recovers 0.12 bits
mlp0 = node(L[0].mlp)  # recovers 1.09 bits
mlp1 = node(L[1].mlp)  # recovers 0.20 bits
mlp2 = node(L[2].mlp)  # recovers 0.57 bits
h3_2 = node(L[3].head[2])  # recovers 0.39 bits
h3_4 = node(L[3].head[4])  # recovers 1.00 bits
mlp3 = node(L[3].mlp)  # recovers 0.35 bits

edges(
    embed >> h0_1,
    embed >> mlp0,
    embed >> mlp1,
    embed >> mlp2,
    embed >> h3_2,
    embed >> h3_4,
    embed >> mlp3,
    h0_1 >> mlp0,
    h0_1 >> mlp1,
    h0_1 >> mlp2,
    h0_1 >> h3_2,
    h0_1 >> h3_4,
    h0_1 >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> mlp2,
    mlp0 >> h3_2,
    mlp0 >> h3_4,
    mlp0 >> mlp3,
    mlp1 >> mlp2,
    mlp1 >> h3_2,
    mlp1 >> h3_4,
    mlp1 >> mlp3,
    mlp2 >> h3_2,
    mlp2 >> h3_4,
    mlp2 >> mlp3,
    h3_2 >> mlp3,
    h3_4 >> mlp3,
    h0_1 >> logits,
    mlp0 >> logits,
    mlp1 >> logits,
    mlp2 >> logits,
    h3_2 >> logits,
    h3_4 >> logits,
    mlp3 >> logits,
)


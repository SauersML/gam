"""Behavior pronoun_gender.because (vpd4l; M's top token is right on 97% of the targets): Gendered
pronoun: the next word is the pronoun for the named person; the counterfactual swaps the name's
gender.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 2.03 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 2.02, L3.H4 1.90, layer 2 MLP 1.30, L3.H2 0.76, layer 1 MLP 0.39 bits.
The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 2.02 bits
mlp1 = node(L[1].mlp)  # recovers 0.39 bits
mlp2 = node(L[2].mlp)  # recovers 1.30 bits
h3_2 = node(L[3].head[2])  # recovers 0.76 bits
h3_4 = node(L[3].head[4])  # recovers 1.90 bits

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> mlp2,
    embed >> h3_2,
    embed >> h3_4,
    mlp0 >> mlp1,
    mlp0 >> mlp2,
    mlp0 >> h3_2,
    mlp0 >> h3_4,
    mlp1 >> mlp2,
    mlp1 >> h3_2,
    mlp1 >> h3_4,
    mlp2 >> h3_2,
    mlp2 >> h3_4,
    mlp0 >> logits,
    mlp1 >> logits,
    mlp2 >> logits,
    h3_2 >> logits,
    h3_4 >> logits,
)


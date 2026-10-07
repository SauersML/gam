"""Behavior induction_phrase.sentence (vpd4l; M's top token is right on 98% of the targets): Induction
on a repeated sentence: a sentence with random nouns in its slots is repeated after a filler
sentence; at a slot the next word is the noun from the first copy.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 12.14 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 12.13, L2.H4 9.76, layer 3 MLP 7.21, L2.H3 5.40, L3.H4 4.92, L3.H5 3.23,
L3.H0 1.46 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 12.13 bits
h2_3 = node(L[2].head[3])  # recovers 5.40 bits
h2_4 = node(L[2].head[4])  # recovers 9.76 bits
h3_0 = node(L[3].head[0])  # recovers 1.46 bits
h3_4 = node(L[3].head[4])  # recovers 4.92 bits
h3_5 = node(L[3].head[5])  # recovers 3.23 bits
mlp3 = node(L[3].mlp)  # recovers 7.21 bits

edges(
    embed >> mlp0,
    embed >> h2_3,
    embed >> h2_4,
    embed >> h3_0,
    embed >> h3_4,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> h3_0,
    mlp0 >> h3_4,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    h2_3 >> h3_0,
    h2_3 >> h3_4,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    h2_4 >> h3_0,
    h2_4 >> h3_4,
    h2_4 >> h3_5,
    h2_4 >> mlp3,
    h3_0 >> mlp3,
    h3_4 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    h3_0 >> logits,
    h3_4 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)


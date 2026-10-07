"""Behavior bigram.mixed (vpd4l; M's top token is right on 70% of the targets): Frequent bigram: the
second word of a common multiword name follows its first word. Phrasings: 'Last year I read a long
article about {X}' | 'Topics: weather, sports, {X}'.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 12.42 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 12.39, layer 3 MLP 5.78, layer 2 MLP 5.08, layer 1 MLP 2.22, L3.H4 2.12,
L2.H3 1.59, L3.H5 1.50, L3.H1 1.38 bits. The program lets every write among them reach every later
read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 12.39 bits
mlp1 = node(L[1].mlp)  # recovers 2.22 bits
h2_3 = node(L[2].head[3])  # recovers 1.59 bits
mlp2 = node(L[2].mlp)  # recovers 5.08 bits
h3_1 = node(L[3].head[1])  # recovers 1.38 bits
h3_4 = node(L[3].head[4])  # recovers 2.12 bits
h3_5 = node(L[3].head[5])  # recovers 1.50 bits
mlp3 = node(L[3].mlp)  # recovers 5.78 bits

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_3,
    embed >> mlp2,
    embed >> h3_1,
    embed >> h3_4,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_3,
    mlp0 >> mlp2,
    mlp0 >> h3_1,
    mlp0 >> h3_4,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    mlp1 >> h2_3,
    mlp1 >> mlp2,
    mlp1 >> h3_1,
    mlp1 >> h3_4,
    mlp1 >> h3_5,
    mlp1 >> mlp3,
    h2_3 >> mlp2,
    h2_3 >> h3_1,
    h2_3 >> h3_4,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    mlp2 >> h3_1,
    mlp2 >> h3_4,
    mlp2 >> h3_5,
    mlp2 >> mlp3,
    h3_1 >> mlp3,
    h3_4 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_3 >> logits,
    mlp2 >> logits,
    h3_1 >> logits,
    h3_4 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)


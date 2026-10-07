"""Behavior capital_to_country.mixed (vpd4l; M's top token is right on 12% of the targets): Factual
recall in reverse: the country whose capital is the named city. Phrasings: '{X} is the capital city
of' | 'Q: Which country has {X} as its capital?\nA:'.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 3.16 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 3.14, L2.H3 2.49, layer 3 MLP 1.21, L3.H4 0.99, layer 1 MLP 0.68, L3.H5
0.64, L3.H0 0.56, layer 2 MLP 0.41 bits. The program lets every write among them reach every later
read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 3.14 bits
mlp1 = node(L[1].mlp)  # recovers 0.68 bits
h2_3 = node(L[2].head[3])  # recovers 2.49 bits
mlp2 = node(L[2].mlp)  # recovers 0.41 bits
h3_0 = node(L[3].head[0])  # recovers 0.56 bits
h3_4 = node(L[3].head[4])  # recovers 0.99 bits
h3_5 = node(L[3].head[5])  # recovers 0.64 bits
mlp3 = node(L[3].mlp)  # recovers 1.21 bits

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_3,
    embed >> mlp2,
    embed >> h3_0,
    embed >> h3_4,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_3,
    mlp0 >> mlp2,
    mlp0 >> h3_0,
    mlp0 >> h3_4,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    mlp1 >> h2_3,
    mlp1 >> mlp2,
    mlp1 >> h3_0,
    mlp1 >> h3_4,
    mlp1 >> h3_5,
    mlp1 >> mlp3,
    h2_3 >> mlp2,
    h2_3 >> h3_0,
    h2_3 >> h3_4,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    mlp2 >> h3_0,
    mlp2 >> h3_4,
    mlp2 >> h3_5,
    mlp2 >> mlp3,
    h3_0 >> mlp3,
    h3_4 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_3 >> logits,
    mlp2 >> logits,
    h3_0 >> logits,
    h3_4 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)


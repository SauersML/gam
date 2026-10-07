"""Behavior country_language.mixed (vpd4l; M's top token is right on 33% of the targets): Factual
recall: the main language of a named country. Phrasings: 'In {X}, most people speak' | 'The official
language of {X} is'.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 3.04 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 3.02, layer 3 MLP 1.62, layer 2 MLP 1.35, L2.H3 0.87, L3.H5 0.60, layer 1
MLP 0.58 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 3.02 bits
mlp1 = node(L[1].mlp)  # recovers 0.58 bits
h2_3 = node(L[2].head[3])  # recovers 0.87 bits
mlp2 = node(L[2].mlp)  # recovers 1.35 bits
h3_5 = node(L[3].head[5])  # recovers 0.60 bits
mlp3 = node(L[3].mlp)  # recovers 1.62 bits

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_3,
    embed >> mlp2,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_3,
    mlp0 >> mlp2,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    mlp1 >> h2_3,
    mlp1 >> mlp2,
    mlp1 >> h3_5,
    mlp1 >> mlp3,
    h2_3 >> mlp2,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    mlp2 >> h3_5,
    mlp2 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_3 >> logits,
    mlp2 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)


"""Behavior demonym.mixed (vpd4l; M's top token is right on 24% of the targets): Factual recall: the
nationality adjective of a named country. Phrasings: 'She was born and raised in {X}, so she is' |
'A citizen of {X} is called'.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 3.27 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 3.26, layer 3 MLP 1.67, layer 2 MLP 1.61, layer 1 MLP 0.70, L2.H3 0.48,
L3.H5 0.43, L3.H1 0.42 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 3.26 bits
mlp1 = node(L[1].mlp)  # recovers 0.70 bits
h2_3 = node(L[2].head[3])  # recovers 0.48 bits
mlp2 = node(L[2].mlp)  # recovers 1.61 bits
h3_1 = node(L[3].head[1])  # recovers 0.42 bits
h3_5 = node(L[3].head[5])  # recovers 0.43 bits
mlp3 = node(L[3].mlp)  # recovers 1.67 bits

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_3,
    embed >> mlp2,
    embed >> h3_1,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_3,
    mlp0 >> mlp2,
    mlp0 >> h3_1,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    mlp1 >> h2_3,
    mlp1 >> mlp2,
    mlp1 >> h3_1,
    mlp1 >> h3_5,
    mlp1 >> mlp3,
    h2_3 >> mlp2,
    h2_3 >> h3_1,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    mlp2 >> h3_1,
    mlp2 >> h3_5,
    mlp2 >> mlp3,
    h3_1 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_3 >> logits,
    mlp2 >> logits,
    h3_1 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)


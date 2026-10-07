"""Behavior possessive_pronoun.lost (vpd4l; M's top token is right on 30% of the targets): Possessive
pronoun: after a named person loses something, 'her' or 'his' follows by the name's usual gender;
the counterfactual swaps the name's gender.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 0.79 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 0.79, L3.H4 0.72, layer 2 MLP 0.42, L3.H2 0.20, layer 1 MLP 0.16, layer 3
MLP 0.10 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 0.79 bits
mlp1 = node(L[1].mlp)  # recovers 0.16 bits
mlp2 = node(L[2].mlp)  # recovers 0.42 bits
h3_2 = node(L[3].head[2])  # recovers 0.20 bits
h3_4 = node(L[3].head[4])  # recovers 0.72 bits
mlp3 = node(L[3].mlp)  # recovers 0.10 bits

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> mlp2,
    embed >> h3_2,
    embed >> h3_4,
    embed >> mlp3,
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
    mlp0 >> logits,
    mlp1 >> logits,
    mlp2 >> logits,
    h3_2 >> logits,
    h3_4 >> logits,
    mlp3 >> logits,
)


"""Behavior bracket_type.mixed (vpd4l; M's top token is right on 55% of the targets): Bracket type
matching: after a run of opened brackets of mixed types and one inner pair closed, the next token
closes the innermost open bracket with its own type; the counterfactual changes that bracket's type.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 5.31 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 5.29, L2.H1 4.15, L3.H3 3.52, layer 1 MLP 2.57, L3.H1 2.15, layer 3 MLP
2.00, L0.H1 1.31, layer 2 MLP 0.90 bits. The program lets every write among them reach every later
read.
"""
from mech import node, edges, L, embed, logits

h0_1 = node(L[0].head[1])  # recovers 1.31 bits
mlp0 = node(L[0].mlp)  # recovers 5.29 bits
mlp1 = node(L[1].mlp)  # recovers 2.57 bits
h2_1 = node(L[2].head[1])  # recovers 4.15 bits
mlp2 = node(L[2].mlp)  # recovers 0.90 bits
h3_1 = node(L[3].head[1])  # recovers 2.15 bits
h3_3 = node(L[3].head[3])  # recovers 3.52 bits
mlp3 = node(L[3].mlp)  # recovers 2.00 bits

edges(
    embed >> h0_1,
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_1,
    embed >> mlp2,
    embed >> h3_1,
    embed >> h3_3,
    embed >> mlp3,
    h0_1 >> mlp0,
    h0_1 >> mlp1,
    h0_1 >> h2_1,
    h0_1 >> mlp2,
    h0_1 >> h3_1,
    h0_1 >> h3_3,
    h0_1 >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_1,
    mlp0 >> mlp2,
    mlp0 >> h3_1,
    mlp0 >> h3_3,
    mlp0 >> mlp3,
    mlp1 >> h2_1,
    mlp1 >> mlp2,
    mlp1 >> h3_1,
    mlp1 >> h3_3,
    mlp1 >> mlp3,
    h2_1 >> mlp2,
    h2_1 >> h3_1,
    h2_1 >> h3_3,
    h2_1 >> mlp3,
    mlp2 >> h3_1,
    mlp2 >> h3_3,
    mlp2 >> mlp3,
    h3_1 >> mlp3,
    h3_3 >> mlp3,
    h0_1 >> logits,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_1 >> logits,
    mlp2 >> logits,
    h3_1 >> logits,
    h3_3 >> logits,
    mlp3 >> logits,
)


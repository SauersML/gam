"""Behavior docstring_args.args (vpd4l; M's top token is right on 90% of the targets): Docstring
argument recall: after two documented arguments, the next documented name is the third argument of
the signature.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 10.71 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 10.70, L3.H4 10.27, layer 3 MLP 7.13, L2.H3 3.76, layer 2 MLP 2.13, L3.H5
1.73, layer 1 MLP 1.67 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 10.70 bits
mlp1 = node(L[1].mlp)  # recovers 1.67 bits
h2_3 = node(L[2].head[3])  # recovers 3.76 bits
mlp2 = node(L[2].mlp)  # recovers 2.13 bits
h3_4 = node(L[3].head[4])  # recovers 10.27 bits
h3_5 = node(L[3].head[5])  # recovers 1.73 bits
mlp3 = node(L[3].mlp)  # recovers 7.13 bits

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_3,
    embed >> mlp2,
    embed >> h3_4,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_3,
    mlp0 >> mlp2,
    mlp0 >> h3_4,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    mlp1 >> h2_3,
    mlp1 >> mlp2,
    mlp1 >> h3_4,
    mlp1 >> h3_5,
    mlp1 >> mlp3,
    h2_3 >> mlp2,
    h2_3 >> h3_4,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    mlp2 >> h3_4,
    mlp2 >> h3_5,
    mlp2 >> mlp3,
    h3_4 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_3 >> logits,
    mlp2 >> logits,
    h3_4 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)


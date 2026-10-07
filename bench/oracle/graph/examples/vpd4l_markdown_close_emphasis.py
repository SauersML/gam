"""Behavior markdown_close.emphasis (vpd4l; M's top token is right on 84% of the targets): Markdown
closing: after a word opened with ** (bold) or _ (italic), the next token closes the same marker;
the counterfactual opens the other marker.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 4.88 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 4.87, L2.H1 3.42, layer 1 MLP 1.80, L3.H3 1.65, L2.H2 1.42, L3.H0 1.19,
layer 2 MLP 1.17, L1.H4 0.66, L0.H3 0.54 bits. The program lets every write among them reach every
later read.
"""
from mech import node, edges, L, embed, logits

h0_3 = node(L[0].head[3])  # recovers 0.54 bits
mlp0 = node(L[0].mlp)  # recovers 4.87 bits
h1_4 = node(L[1].head[4])  # recovers 0.66 bits
mlp1 = node(L[1].mlp)  # recovers 1.80 bits
h2_1 = node(L[2].head[1])  # recovers 3.42 bits
h2_2 = node(L[2].head[2])  # recovers 1.42 bits
mlp2 = node(L[2].mlp)  # recovers 1.17 bits
h3_0 = node(L[3].head[0])  # recovers 1.19 bits
h3_3 = node(L[3].head[3])  # recovers 1.65 bits

edges(
    embed >> h0_3,
    embed >> mlp0,
    embed >> h1_4,
    embed >> mlp1,
    embed >> h2_1,
    embed >> h2_2,
    embed >> mlp2,
    embed >> h3_0,
    embed >> h3_3,
    h0_3 >> mlp0,
    h0_3 >> h1_4,
    h0_3 >> mlp1,
    h0_3 >> h2_1,
    h0_3 >> h2_2,
    h0_3 >> mlp2,
    h0_3 >> h3_0,
    h0_3 >> h3_3,
    mlp0 >> h1_4,
    mlp0 >> mlp1,
    mlp0 >> h2_1,
    mlp0 >> h2_2,
    mlp0 >> mlp2,
    mlp0 >> h3_0,
    mlp0 >> h3_3,
    h1_4 >> mlp1,
    h1_4 >> h2_1,
    h1_4 >> h2_2,
    h1_4 >> mlp2,
    h1_4 >> h3_0,
    h1_4 >> h3_3,
    mlp1 >> h2_1,
    mlp1 >> h2_2,
    mlp1 >> mlp2,
    mlp1 >> h3_0,
    mlp1 >> h3_3,
    h2_1 >> mlp2,
    h2_1 >> h3_0,
    h2_1 >> h3_3,
    h2_2 >> mlp2,
    h2_2 >> h3_0,
    h2_2 >> h3_3,
    mlp2 >> h3_0,
    mlp2 >> h3_3,
    h0_3 >> logits,
    mlp0 >> logits,
    h1_4 >> logits,
    mlp1 >> logits,
    h2_1 >> logits,
    h2_2 >> logits,
    mlp2 >> logits,
    h3_0 >> logits,
    h3_3 >> logits,
)


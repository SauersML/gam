"""Behavior gerund.mixed (vpd4l; M's top token is right on 22% of the targets): Morphology: the -ing
form of a verb, with its spelling changes. Phrasings: 'go: going\nsit: sitting\n{X}:' | 'I like to
{X}. Right now I am'.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 1.06 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 1.05, L3.H4 0.65, layer 3 MLP 0.58, layer 1 MLP 0.42, layer 2 MLP 0.36,
L0.H3 0.25, L2.H4 0.23, L1.H1 0.22, L2.H3 0.20, L2.H2 0.20, L1.H2 0.16, L2.H1 0.16, L3.H0 0.16,
L3.H1 0.12 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

h0_3 = node(L[0].head[3])  # recovers 0.25 bits
mlp0 = node(L[0].mlp)  # recovers 1.05 bits
h1_1 = node(L[1].head[1])  # recovers 0.22 bits
h1_2 = node(L[1].head[2])  # recovers 0.16 bits
mlp1 = node(L[1].mlp)  # recovers 0.42 bits
h2_1 = node(L[2].head[1])  # recovers 0.16 bits
h2_2 = node(L[2].head[2])  # recovers 0.20 bits
h2_3 = node(L[2].head[3])  # recovers 0.20 bits
h2_4 = node(L[2].head[4])  # recovers 0.23 bits
mlp2 = node(L[2].mlp)  # recovers 0.36 bits
h3_0 = node(L[3].head[0])  # recovers 0.16 bits
h3_1 = node(L[3].head[1])  # recovers 0.12 bits
h3_4 = node(L[3].head[4])  # recovers 0.65 bits
mlp3 = node(L[3].mlp)  # recovers 0.58 bits

edges(
    embed >> h0_3,
    embed >> mlp0,
    embed >> h1_1,
    embed >> h1_2,
    embed >> mlp1,
    embed >> h2_1,
    embed >> h2_2,
    embed >> h2_3,
    embed >> h2_4,
    embed >> mlp2,
    embed >> h3_0,
    embed >> h3_1,
    embed >> h3_4,
    embed >> mlp3,
    h0_3 >> mlp0,
    h0_3 >> h1_1,
    h0_3 >> h1_2,
    h0_3 >> mlp1,
    h0_3 >> h2_1,
    h0_3 >> h2_2,
    h0_3 >> h2_3,
    h0_3 >> h2_4,
    h0_3 >> mlp2,
    h0_3 >> h3_0,
    h0_3 >> h3_1,
    h0_3 >> h3_4,
    h0_3 >> mlp3,
    mlp0 >> h1_1,
    mlp0 >> h1_2,
    mlp0 >> mlp1,
    mlp0 >> h2_1,
    mlp0 >> h2_2,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> mlp2,
    mlp0 >> h3_0,
    mlp0 >> h3_1,
    mlp0 >> h3_4,
    mlp0 >> mlp3,
    h1_1 >> mlp1,
    h1_1 >> h2_1,
    h1_1 >> h2_2,
    h1_1 >> h2_3,
    h1_1 >> h2_4,
    h1_1 >> mlp2,
    h1_1 >> h3_0,
    h1_1 >> h3_1,
    h1_1 >> h3_4,
    h1_1 >> mlp3,
    h1_2 >> mlp1,
    h1_2 >> h2_1,
    h1_2 >> h2_2,
    h1_2 >> h2_3,
    h1_2 >> h2_4,
    h1_2 >> mlp2,
    h1_2 >> h3_0,
    h1_2 >> h3_1,
    h1_2 >> h3_4,
    h1_2 >> mlp3,
    mlp1 >> h2_1,
    mlp1 >> h2_2,
    mlp1 >> h2_3,
    mlp1 >> h2_4,
    mlp1 >> mlp2,
    mlp1 >> h3_0,
    mlp1 >> h3_1,
    mlp1 >> h3_4,
    mlp1 >> mlp3,
    h2_1 >> mlp2,
    h2_1 >> h3_0,
    h2_1 >> h3_1,
    h2_1 >> h3_4,
    h2_1 >> mlp3,
    h2_2 >> mlp2,
    h2_2 >> h3_0,
    h2_2 >> h3_1,
    h2_2 >> h3_4,
    h2_2 >> mlp3,
    h2_3 >> mlp2,
    h2_3 >> h3_0,
    h2_3 >> h3_1,
    h2_3 >> h3_4,
    h2_3 >> mlp3,
    h2_4 >> mlp2,
    h2_4 >> h3_0,
    h2_4 >> h3_1,
    h2_4 >> h3_4,
    h2_4 >> mlp3,
    mlp2 >> h3_0,
    mlp2 >> h3_1,
    mlp2 >> h3_4,
    mlp2 >> mlp3,
    h3_0 >> mlp3,
    h3_1 >> mlp3,
    h3_4 >> mlp3,
    h0_3 >> logits,
    mlp0 >> logits,
    h1_1 >> logits,
    h1_2 >> logits,
    mlp1 >> logits,
    h2_1 >> logits,
    h2_2 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    mlp2 >> logits,
    h3_0 >> logits,
    h3_1 >> logits,
    h3_4 >> logits,
    mlp3 >> logits,
)


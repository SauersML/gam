"""Behavior alphabet.upper (qwen3-0.6b; M's top token is right on 78% of the targets): Alphabet
succession: a run of capital letters; the next letter continues it.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 5.75 bits per target token between the clean and the counterfactual next-token
distributions, and more than their opaque price per token (1/2 log2 N bits per weight, N = 2^24):
0.28 for a head, 6.75 for a whole MLP; an MLP may instead enter as the neurons that pay for
themselves: L21.H6 4.65, L11.H9 3.76, L0.H3 2.47, L0.H12 1.84, L12.H1 1.27, L22.H8 1.21, L25.H8
1.03, L23.H6 0.98, L25.H9 0.96, L27.H15 0.73, L5.H10 0.68 bits. The program lets every write among
them reach every later read.
"""
from mech import node, edges, L, embed, logits

h0_3 = node(L[0].head[3])  # recovers 2.47 bits
h0_12 = node(L[0].head[12])  # recovers 1.84 bits
h5_10 = node(L[5].head[10])  # recovers 0.68 bits
h11_9 = node(L[11].head[9])  # recovers 3.76 bits
h12_1 = node(L[12].head[1])  # recovers 1.27 bits
h21_6 = node(L[21].head[6])  # recovers 4.65 bits
h22_8 = node(L[22].head[8])  # recovers 1.21 bits
h23_6 = node(L[23].head[6])  # recovers 0.98 bits
h25_8 = node(L[25].head[8])  # recovers 1.03 bits
h25_9 = node(L[25].head[9])  # recovers 0.96 bits
h27_15 = node(L[27].head[15])  # recovers 0.73 bits

edges(
    embed >> h0_3,
    embed >> h0_12,
    embed >> h5_10,
    embed >> h11_9,
    embed >> h12_1,
    embed >> h21_6,
    embed >> h22_8,
    embed >> h23_6,
    embed >> h25_8,
    embed >> h25_9,
    embed >> h27_15,
    h0_3 >> h5_10,
    h0_3 >> h11_9,
    h0_3 >> h12_1,
    h0_3 >> h21_6,
    h0_3 >> h22_8,
    h0_3 >> h23_6,
    h0_3 >> h25_8,
    h0_3 >> h25_9,
    h0_3 >> h27_15,
    h0_12 >> h5_10,
    h0_12 >> h11_9,
    h0_12 >> h12_1,
    h0_12 >> h21_6,
    h0_12 >> h22_8,
    h0_12 >> h23_6,
    h0_12 >> h25_8,
    h0_12 >> h25_9,
    h0_12 >> h27_15,
    h5_10 >> h11_9,
    h5_10 >> h12_1,
    h5_10 >> h21_6,
    h5_10 >> h22_8,
    h5_10 >> h23_6,
    h5_10 >> h25_8,
    h5_10 >> h25_9,
    h5_10 >> h27_15,
    h11_9 >> h12_1,
    h11_9 >> h21_6,
    h11_9 >> h22_8,
    h11_9 >> h23_6,
    h11_9 >> h25_8,
    h11_9 >> h25_9,
    h11_9 >> h27_15,
    h12_1 >> h21_6,
    h12_1 >> h22_8,
    h12_1 >> h23_6,
    h12_1 >> h25_8,
    h12_1 >> h25_9,
    h12_1 >> h27_15,
    h21_6 >> h22_8,
    h21_6 >> h23_6,
    h21_6 >> h25_8,
    h21_6 >> h25_9,
    h21_6 >> h27_15,
    h22_8 >> h23_6,
    h22_8 >> h25_8,
    h22_8 >> h25_9,
    h22_8 >> h27_15,
    h23_6 >> h25_8,
    h23_6 >> h25_9,
    h23_6 >> h27_15,
    h25_8 >> h27_15,
    h25_9 >> h27_15,
    h0_3 >> logits,
    h0_12 >> logits,
    h5_10 >> logits,
    h11_9 >> logits,
    h12_1 >> logits,
    h21_6 >> logits,
    h22_8 >> logits,
    h23_6 >> logits,
    h25_8 >> logits,
    h25_9 >> logits,
    h27_15 >> logits,
)


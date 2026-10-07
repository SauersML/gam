"""Behavior greater_than.war (qwen3-0.6b; M's top token is right on 100% of the targets): Greater-than:
the end year starts with the same century, so its last two digits must exceed the start year's.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 3.75 bits per target token between the clean and the counterfactual next-token
distributions, and more than their opaque price per token (1/2 log2 N bits per weight, N = 2^24):
0.28 for a head, 6.75 for a whole MLP; an MLP may instead enter as the neurons that pay for
themselves: L0.H7 2.54, L0.H3 1.95, L11.H9 0.96, L21.H9 0.88, L20.H14 0.85, L25.H9 0.83, L20.H3
0.72, L20.H15 0.54, L21.H6 0.38 bits. The program lets every write among them reach every later
read.
"""
from mech import node, edges, L, embed, logits

h0_3 = node(L[0].head[3])  # recovers 1.95 bits
h0_7 = node(L[0].head[7])  # recovers 2.54 bits
h11_9 = node(L[11].head[9])  # recovers 0.96 bits
h20_3 = node(L[20].head[3])  # recovers 0.72 bits
h20_14 = node(L[20].head[14])  # recovers 0.85 bits
h20_15 = node(L[20].head[15])  # recovers 0.54 bits
h21_6 = node(L[21].head[6])  # recovers 0.38 bits
h21_9 = node(L[21].head[9])  # recovers 0.88 bits
h25_9 = node(L[25].head[9])  # recovers 0.83 bits

edges(
    embed >> h0_3,
    embed >> h0_7,
    embed >> h11_9,
    embed >> h20_3,
    embed >> h20_14,
    embed >> h20_15,
    embed >> h21_6,
    embed >> h21_9,
    embed >> h25_9,
    h0_3 >> h11_9,
    h0_3 >> h20_3,
    h0_3 >> h20_14,
    h0_3 >> h20_15,
    h0_3 >> h21_6,
    h0_3 >> h21_9,
    h0_3 >> h25_9,
    h0_7 >> h11_9,
    h0_7 >> h20_3,
    h0_7 >> h20_14,
    h0_7 >> h20_15,
    h0_7 >> h21_6,
    h0_7 >> h21_9,
    h0_7 >> h25_9,
    h11_9 >> h20_3,
    h11_9 >> h20_14,
    h11_9 >> h20_15,
    h11_9 >> h21_6,
    h11_9 >> h21_9,
    h11_9 >> h25_9,
    h20_3 >> h21_6,
    h20_3 >> h21_9,
    h20_3 >> h25_9,
    h20_14 >> h21_6,
    h20_14 >> h21_9,
    h20_14 >> h25_9,
    h20_15 >> h21_6,
    h20_15 >> h21_9,
    h20_15 >> h25_9,
    h21_6 >> h25_9,
    h21_9 >> h25_9,
    h0_3 >> logits,
    h0_7 >> logits,
    h11_9 >> logits,
    h20_3 >> logits,
    h20_14 >> logits,
    h20_15 >> logits,
    h21_6 >> logits,
    h21_9 >> logits,
    h25_9 >> logits,
)


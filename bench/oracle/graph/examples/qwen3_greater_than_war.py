"""Behavior greater_than.war (qwen3-0.6b; M's top token is right on 100% of the targets): Greater-than:
the end year starts with the same century, so its last two digits must exceed the start year's.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 3.75 bits per target token between the clean and the counterfactual next-token
distributions, and more than their opaque price per token (1/2 log2 N bits per weight, N = 2^24):
0.28 for a head, 6.75 for a whole MLP; an MLP may instead enter as the neurons that pay for
themselves: L0.H7 2.54, L0.H3 1.95, layer 18 transcoder features (summed) 1.75, layer 21 transcoder
features (summed) 1.73, layer 20 transcoder features (summed) 1.31, layer 22 transcoder features
(summed) 0.97, L11.H9 0.96, L21.H9 0.88, L20.H14 0.85, L25.H9 0.83, layer 26 transcoder features
(summed) 0.74, L20.H3 0.72, L20.H15 0.54, L21.H6 0.38, layer 16 transcoder features (summed) 0.21
bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, PD, embed, logits

h0_3 = node(L[0].head[3])  # recovers 1.95 bits
h0_7 = node(L[0].head[7])  # recovers 2.54 bits
h11_9 = node(L[11].head[9])  # recovers 0.96 bits
# layer 16's MLP as the transcoder features that recover more than their price when patched alone (0.21 bits summed)
tc16 = node(PD.tc[16][12193, 36503, 37058, 55487, 64985, 72946, 102563, 111834, 130371, 137545, 146528, 157329, 160941])
# layer 18's MLP as the transcoder features that recover more than their price when patched alone (1.75 bits summed)
tc18 = node(PD.tc[18][2067, 3832, 33104, 36429, 43828, 47651, 48600, 58370, 66622, 74641, 87899, 93523, 96642, 120742, 149188, 157236])
h20_3 = node(L[20].head[3])  # recovers 0.72 bits
h20_14 = node(L[20].head[14])  # recovers 0.85 bits
h20_15 = node(L[20].head[15])  # recovers 0.54 bits
# layer 20's MLP as the transcoder features that recover more than their price when patched alone (1.31 bits summed)
tc20 = node(PD.tc[20][2446, 5775, 8800, 20075, 44766, 64880, 67534, 68965, 71765, 72908, 86874, 102647, 102733, 114244, 116177, 118133, 118822, 125363, 128372, 130292, 142292, 144554, 155580, 156307, 157873, 161858])
h21_6 = node(L[21].head[6])  # recovers 0.38 bits
h21_9 = node(L[21].head[9])  # recovers 0.88 bits
# layer 21's MLP as the transcoder features that recover more than their price when patched alone (1.73 bits summed)
tc21 = node(PD.tc[21][11016, 16820, 17921, 88162, 91314, 96504, 107839, 122540, 128820, 129568, 136370, 144983, 149914, 155497, 162757])
# layer 22's MLP as the transcoder features that recover more than their price when patched alone (0.97 bits summed)
tc22 = node(PD.tc[22][1756, 4201, 4432, 17771, 19980, 25676, 30988, 93382, 103181, 114871, 123825, 132757, 158006])
h25_9 = node(L[25].head[9])  # recovers 0.83 bits
# layer 26's MLP as the transcoder features that recover more than their price when patched alone (0.74 bits summed)
tc26 = node(PD.tc[26][24064, 38588, 64566, 95593, 96541, 102542, 105337, 106122, 147197, 153256, 156482])

edges(
    embed >> h0_3,
    embed >> h0_7,
    embed >> h11_9,
    embed >> tc16,
    embed >> tc18,
    embed >> h20_3,
    embed >> h20_14,
    embed >> h20_15,
    embed >> tc20,
    embed >> h21_6,
    embed >> h21_9,
    embed >> tc21,
    embed >> tc22,
    embed >> h25_9,
    embed >> tc26,
    h0_3 >> h11_9,
    h0_3 >> tc16,
    h0_3 >> tc18,
    h0_3 >> h20_3,
    h0_3 >> h20_14,
    h0_3 >> h20_15,
    h0_3 >> tc20,
    h0_3 >> h21_6,
    h0_3 >> h21_9,
    h0_3 >> tc21,
    h0_3 >> tc22,
    h0_3 >> h25_9,
    h0_3 >> tc26,
    h0_7 >> h11_9,
    h0_7 >> tc16,
    h0_7 >> tc18,
    h0_7 >> h20_3,
    h0_7 >> h20_14,
    h0_7 >> h20_15,
    h0_7 >> tc20,
    h0_7 >> h21_6,
    h0_7 >> h21_9,
    h0_7 >> tc21,
    h0_7 >> tc22,
    h0_7 >> h25_9,
    h0_7 >> tc26,
    h11_9 >> tc16,
    h11_9 >> tc18,
    h11_9 >> h20_3,
    h11_9 >> h20_14,
    h11_9 >> h20_15,
    h11_9 >> tc20,
    h11_9 >> h21_6,
    h11_9 >> h21_9,
    h11_9 >> tc21,
    h11_9 >> tc22,
    h11_9 >> h25_9,
    h11_9 >> tc26,
    tc16 >> tc18,
    tc16 >> h20_3,
    tc16 >> h20_14,
    tc16 >> h20_15,
    tc16 >> tc20,
    tc16 >> h21_6,
    tc16 >> h21_9,
    tc16 >> tc21,
    tc16 >> tc22,
    tc16 >> h25_9,
    tc16 >> tc26,
    tc18 >> h20_3,
    tc18 >> h20_14,
    tc18 >> h20_15,
    tc18 >> tc20,
    tc18 >> h21_6,
    tc18 >> h21_9,
    tc18 >> tc21,
    tc18 >> tc22,
    tc18 >> h25_9,
    tc18 >> tc26,
    h20_3 >> tc20,
    h20_3 >> h21_6,
    h20_3 >> h21_9,
    h20_3 >> tc21,
    h20_3 >> tc22,
    h20_3 >> h25_9,
    h20_3 >> tc26,
    h20_14 >> tc20,
    h20_14 >> h21_6,
    h20_14 >> h21_9,
    h20_14 >> tc21,
    h20_14 >> tc22,
    h20_14 >> h25_9,
    h20_14 >> tc26,
    h20_15 >> tc20,
    h20_15 >> h21_6,
    h20_15 >> h21_9,
    h20_15 >> tc21,
    h20_15 >> tc22,
    h20_15 >> h25_9,
    h20_15 >> tc26,
    tc20 >> h21_6,
    tc20 >> h21_9,
    tc20 >> tc21,
    tc20 >> tc22,
    tc20 >> h25_9,
    tc20 >> tc26,
    h21_6 >> tc21,
    h21_6 >> tc22,
    h21_6 >> h25_9,
    h21_6 >> tc26,
    h21_9 >> tc21,
    h21_9 >> tc22,
    h21_9 >> h25_9,
    h21_9 >> tc26,
    tc21 >> tc22,
    tc21 >> h25_9,
    tc21 >> tc26,
    tc22 >> h25_9,
    tc22 >> tc26,
    h25_9 >> tc26,
    h0_3 >> logits,
    h0_7 >> logits,
    h11_9 >> logits,
    tc16 >> logits,
    tc18 >> logits,
    h20_3 >> logits,
    h20_14 >> logits,
    h20_15 >> logits,
    tc20 >> logits,
    h21_6 >> logits,
    h21_9 >> logits,
    tc21 >> logits,
    tc22 >> logits,
    h25_9 >> logits,
    tc26 >> logits,
)


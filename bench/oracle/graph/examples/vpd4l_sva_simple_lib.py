"""Behavior sva.simple (vpd4l, our library: decomp's exact all-on start, arm grouped_own; M's top token
is right on 7% of the targets): Subject-verb agreement after a determiner and noun; the
counterfactual flips the subject's number.

Nodes: per layer's attention and MLP, the library parts whose clean contribution (all their VPD
slices), patched into the counterfactual run, recovers the most of the 2.52 bits per target, as many
as pay for their opaque price: layer 0 attn: 4 parts leaving 2.48 bits; layer 0 mlp: 16 parts
leaving 0.47 bits; layer 1 attn: 8 parts leaving 2.49 bits; layer 2 attn: 16 parts leaving 2.33
bits; layer 2 mlp: 8 parts leaving 0.71 bits; layer 3 attn: 8 parts leaving 1.03 bits; layer 3 mlp:
4 parts leaving 1.63 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, PD, embed, logits

lib_attn0 = node(PD.lib[0].attn[
    9, 82, 284, 467
])
lib_mlp0 = node(PD.lib[0].mlp[
    25, 64, 149, 188, 485, 543, 548, 568, 588, 772, 787, 791, 978, 1162, 1330, 1426
])
lib_attn1 = node(PD.lib[1].attn[
    99, 184, 216, 230, 261, 323, 391, 414
])
lib_attn2 = node(PD.lib[2].attn[
    83, 101, 102, 382, 402, 431, 491, 506, 510, 550, 565, 664, 719, 788, 850, 865
])
lib_mlp2 = node(PD.lib[2].mlp[
    14, 143, 155, 197, 373, 446, 458, 506
])
lib_attn3 = node(PD.lib[3].attn[
    18, 56, 262, 313, 341, 418, 539, 543
])
lib_mlp3 = node(PD.lib[3].mlp[
    165, 396, 679, 741
])

edges(
    embed >> lib_attn0,
    embed >> lib_mlp0,
    embed >> lib_attn1,
    embed >> lib_attn2,
    embed >> lib_mlp2,
    embed >> lib_attn3,
    embed >> lib_mlp3,
    lib_attn0 >> lib_mlp0,
    lib_attn0 >> lib_attn1,
    lib_attn0 >> lib_attn2,
    lib_attn0 >> lib_mlp2,
    lib_attn0 >> lib_attn3,
    lib_attn0 >> lib_mlp3,
    lib_mlp0 >> lib_attn1,
    lib_mlp0 >> lib_attn2,
    lib_mlp0 >> lib_mlp2,
    lib_mlp0 >> lib_attn3,
    lib_mlp0 >> lib_mlp3,
    lib_attn1 >> lib_attn2,
    lib_attn1 >> lib_mlp2,
    lib_attn1 >> lib_attn3,
    lib_attn1 >> lib_mlp3,
    lib_attn2 >> lib_mlp2,
    lib_attn2 >> lib_attn3,
    lib_attn2 >> lib_mlp3,
    lib_mlp2 >> lib_attn3,
    lib_mlp2 >> lib_mlp3,
    lib_attn3 >> lib_mlp3,
    lib_attn0 >> logits,
    lib_mlp0 >> logits,
    lib_attn1 >> logits,
    lib_attn2 >> logits,
    lib_mlp2 >> logits,
    lib_attn3 >> logits,
    lib_mlp3 >> logits,
)

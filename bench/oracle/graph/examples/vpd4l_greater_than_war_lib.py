"""Behavior greater_than.war (vpd4l, our library: decomp's exact all-on start, arm grouped_own; M's top
token is right on 96% of the targets): Greater-than: the end year starts with the same century, so
its last two digits must exceed the start year's.

Nodes: per layer's attention and MLP, the library parts whose clean contribution (all their VPD
slices), patched into the counterfactual run, recovers the most of the 2.83 bits per target, as many
as pay for their opaque price: layer 0 attn: 32 parts leaving 2.55 bits; layer 0 mlp: 64 parts
leaving 0.63 bits; layer 1 attn: 8 parts leaving 2.74 bits; layer 1 mlp: 1 parts leaving 2.78 bits;
layer 2 attn: 32 parts leaving 0.45 bits; layer 2 mlp: 8 parts leaving 2.64 bits; layer 3 attn: 128
parts leaving 2.06 bits; layer 3 mlp: 16 parts leaving 1.65 bits. The program lets every write among
them reach every later read.
"""
from mech import node, edges, PD, embed, logits

lib_attn0 = node(PD.lib[0].attn[
    0, 6, 9, 13, 14, 22, 28, 45, 56, 61, 70, 75, 76, 80, 96, 102, 112, 146, 157, 162, 187, 201, 218,
    229, 239, 284, 358, 392, 494, 523, 580, 686
])
lib_mlp0 = node(PD.lib[0].mlp[
    25, 35, 49, 125, 138, 169, 238, 240, 274, 286, 299, 302, 304, 321, 322, 340, 351, 352, 362, 381,
    394, 397, 407, 429, 467, 471, 479, 501, 508, 520, 524, 614, 623, 626, 628, 653, 682, 706, 727,
    741, 747, 791, 802, 853, 971, 980, 983, 998, 1037, 1061, 1064, 1084, 1147, 1181, 1201, 1247,
    1262, 1291, 1330, 1354, 1421, 1426, 1435, 1438
])
lib_attn1 = node(PD.lib[1].attn[
    38, 205, 227, 354, 408, 409, 414, 448
])
lib_mlp1 = node(PD.lib[1].mlp[
    180
])
lib_attn2 = node(PD.lib[2].attn[
    14, 56, 63, 93, 101, 104, 381, 416, 421, 443, 506, 517, 527, 533, 555, 559, 563, 619, 626, 635,
    667, 681, 710, 714, 731, 749, 766, 770, 810, 844, 854, 866
])
lib_mlp2 = node(PD.lib[2].mlp[
    14, 45, 92, 261, 297, 390, 402, 542
])
lib_attn3 = node(PD.lib[3].attn[
    1, 8, 14, 19, 24, 29, 36, 51, 53, 55, 56, 57, 59, 62, 67, 80, 82, 86, 87, 91, 96, 121, 128, 143,
    172, 197, 214, 216, 217, 222, 233, 235, 240, 241, 243, 251, 252, 256, 260, 264, 266, 269, 270,
    272, 280, 289, 292, 294, 298, 302, 307, 309, 312, 315, 316, 321, 332, 334, 341, 344, 345, 349,
    355, 356, 359, 364, 365, 368, 375, 386, 388, 389, 390, 392, 398, 403, 404, 408, 410, 412, 418,
    420, 422, 423, 425, 429, 434, 436, 440, 442, 444, 451, 453, 457, 459, 460, 463, 464, 465, 469,
    471, 478, 479, 482, 484, 485, 487, 491, 494, 495, 499, 500, 502, 507, 508, 509, 511, 512, 513,
    514, 527, 532, 533, 534, 539, 542, 543, 544
])
lib_mlp3 = node(PD.lib[3].mlp[
    65, 151, 192, 208, 343, 387, 390, 489, 523, 591, 628, 854, 898, 968, 1105, 1145
])

edges(
    embed >> lib_attn0,
    embed >> lib_mlp0,
    embed >> lib_attn1,
    embed >> lib_mlp1,
    embed >> lib_attn2,
    embed >> lib_mlp2,
    embed >> lib_attn3,
    embed >> lib_mlp3,
    lib_attn0 >> lib_mlp0,
    lib_attn0 >> lib_attn1,
    lib_attn0 >> lib_mlp1,
    lib_attn0 >> lib_attn2,
    lib_attn0 >> lib_mlp2,
    lib_attn0 >> lib_attn3,
    lib_attn0 >> lib_mlp3,
    lib_mlp0 >> lib_attn1,
    lib_mlp0 >> lib_mlp1,
    lib_mlp0 >> lib_attn2,
    lib_mlp0 >> lib_mlp2,
    lib_mlp0 >> lib_attn3,
    lib_mlp0 >> lib_mlp3,
    lib_attn1 >> lib_mlp1,
    lib_attn1 >> lib_attn2,
    lib_attn1 >> lib_mlp2,
    lib_attn1 >> lib_attn3,
    lib_attn1 >> lib_mlp3,
    lib_mlp1 >> lib_attn2,
    lib_mlp1 >> lib_mlp2,
    lib_mlp1 >> lib_attn3,
    lib_mlp1 >> lib_mlp3,
    lib_attn2 >> lib_mlp2,
    lib_attn2 >> lib_attn3,
    lib_attn2 >> lib_mlp3,
    lib_mlp2 >> lib_attn3,
    lib_mlp2 >> lib_mlp3,
    lib_attn3 >> lib_mlp3,
    lib_attn0 >> logits,
    lib_mlp0 >> logits,
    lib_attn1 >> logits,
    lib_mlp1 >> logits,
    lib_attn2 >> logits,
    lib_mlp2 >> logits,
    lib_attn3 >> logits,
    lib_mlp3 >> logits,
)

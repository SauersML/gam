"""Behavior induction_random.words8 (vpd4l, our library: decomp's exact all-on start, arm grouped_own;
M's top token is right on 98% of the targets): Induction: a list of 8 random words is repeated; at a
point in the repeat the next word is the one that followed the same word in the first copy.

Nodes: per layer's attention and MLP, the library parts whose clean contribution (all their VPD
slices), patched into the counterfactual run, recovers the most of the 7.43 bits per target, as many
as pay for their opaque price: layer 0 attn: 32 parts leaving 7.12 bits; layer 0 mlp: 128 parts
leaving 1.24 bits; layer 1 attn: 4 parts leaving 7.33 bits; layer 1 mlp: 2 parts leaving 7.12 bits;
layer 2 attn: 128 parts leaving 0.37 bits; layer 2 mlp: 1 parts leaving 7.40 bits; layer 3 attn: 256
parts leaving 5.03 bits; layer 3 mlp: 64 parts leaving 4.29 bits. The program lets every write among
them reach every later read.
"""
from mech import node, edges, PD, embed, logits

lib_attn0 = node(PD.lib[0].attn[
    9, 12, 13, 17, 30, 42, 61, 71, 76, 80, 81, 82, 83, 111, 126, 183, 211, 297, 320, 358, 366, 425,
    494, 506, 533, 538, 580, 588, 646, 649, 678, 686
])
lib_mlp0 = node(PD.lib[0].mlp[
    25, 35, 66, 117, 125, 145, 168, 169, 170, 181, 188, 197, 208, 236, 238, 243, 247, 265, 273, 276,
    285, 286, 299, 302, 304, 307, 308, 314, 321, 322, 351, 352, 374, 381, 388, 394, 397, 407, 414,
    425, 429, 453, 467, 479, 485, 501, 503, 520, 533, 551, 573, 588, 590, 606, 607, 614, 616, 623,
    628, 640, 667, 682, 700, 712, 721, 741, 747, 758, 772, 781, 787, 791, 792, 811, 820, 827, 833,
    836, 885, 890, 895, 913, 924, 946, 969, 976, 983, 997, 1015, 1016, 1025, 1037, 1061, 1064, 1077,
    1084, 1093, 1115, 1129, 1147, 1162, 1166, 1181, 1186, 1201, 1207, 1210, 1217, 1236, 1247, 1256,
    1267, 1282, 1298, 1305, 1312, 1330, 1341, 1346, 1364, 1365, 1370, 1372, 1426, 1435, 1438, 1442,
    1445
])
lib_attn1 = node(PD.lib[1].attn[
    38, 293, 315, 344
])
lib_mlp1 = node(PD.lib[1].mlp[
    36, 399
])
lib_attn2 = node(PD.lib[2].attn[
    6, 14, 21, 56, 61, 62, 63, 66, 76, 83, 93, 101, 104, 115, 141, 146, 151, 152, 206, 372, 375,
    378, 381, 382, 383, 391, 394, 410, 416, 421, 422, 425, 438, 440, 445, 452, 457, 462, 468, 475,
    479, 481, 494, 498, 502, 503, 506, 513, 517, 533, 550, 554, 555, 559, 561, 563, 577, 579, 588,
    592, 595, 598, 600, 602, 604, 609, 617, 618, 619, 626, 627, 635, 637, 641, 647, 649, 651, 652,
    654, 657, 660, 667, 675, 679, 681, 682, 687, 699, 710, 714, 717, 722, 725, 728, 731, 743, 744,
    746, 749, 751, 760, 762, 764, 766, 770, 773, 776, 787, 792, 795, 801, 807, 808, 810, 818, 822,
    831, 844, 854, 855, 861, 870, 871, 878, 887, 890, 895, 910
])
lib_mlp2 = node(PD.lib[2].mlp[
    390
])
lib_attn3 = node(PD.lib[3].attn[
    8, 18, 25, 29, 35, 55, 56, 59, 67, 74, 75, 86, 91, 137, 160, 197, 201, 217, 218, 219, 220, 221,
    222, 223, 224, 225, 226, 228, 229, 230, 232, 233, 234, 235, 236, 237, 239, 241, 242, 243, 244,
    245, 248, 249, 250, 251, 252, 255, 256, 258, 259, 260, 261, 263, 264, 265, 266, 268, 269, 270,
    272, 273, 275, 277, 279, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 292, 295, 296, 297,
    298, 299, 301, 304, 306, 307, 309, 310, 312, 313, 315, 316, 317, 318, 319, 320, 324, 329, 330,
    331, 332, 334, 335, 338, 340, 341, 342, 343, 344, 345, 348, 349, 351, 352, 353, 354, 355, 356,
    359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 371, 374, 375, 378, 379, 380, 381, 382,
    383, 384, 385, 387, 388, 389, 390, 391, 392, 394, 397, 398, 399, 401, 402, 403, 407, 408, 410,
    411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 422, 423, 426, 427, 432, 433, 434, 435, 436,
    438, 439, 440, 441, 442, 444, 445, 446, 448, 449, 450, 451, 453, 455, 456, 457, 458, 459, 460,
    461, 462, 463, 464, 466, 467, 468, 469, 470, 472, 474, 477, 478, 479, 480, 481, 482, 484, 485,
    486, 487, 488, 489, 490, 492, 493, 494, 495, 496, 497, 499, 500, 502, 503, 504, 505, 507, 508,
    509, 511, 513, 514, 515, 516, 520, 521, 522, 523, 526, 527, 528, 529, 532, 533, 534, 535, 536,
    537, 539, 540, 542, 543, 546
])
lib_mlp3 = node(PD.lib[3].mlp[
    45, 62, 65, 68, 69, 114, 130, 137, 149, 164, 181, 192, 201, 204, 239, 295, 312, 353, 358, 362,
    368, 375, 408, 409, 423, 432, 437, 459, 461, 516, 523, 542, 554, 576, 594, 600, 605, 634, 644,
    681, 689, 702, 731, 735, 757, 763, 777, 783, 791, 796, 827, 858, 941, 959, 976, 1005, 1035,
    1039, 1046, 1080, 1098, 1143, 1162, 1231
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

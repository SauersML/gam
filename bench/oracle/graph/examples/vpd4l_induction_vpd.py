"""Induction on vpd4l through VPD's subcomponents: when a stretch of text repeats, predict the token
that followed the current token in its first copy ("A B ... A -> B").

Nodes: the attention subcomponents whose removal (W - u_i v_i^T at every token) costs at least 0.05
bits per target token, KL(M || M_e) over the full vocabulary, on repeated Pile text (8 held-out
validation sequences, 48 tokens repeated once, 47 targets each, where M is right 87% of the time).
The numbers in the comments are those costs. One node per layer, because these subcomponents do not
follow head boundaries: most spread their weight over several of the 6 heads.

Layer 1's query 316 and key 329 cost the most (3.8 bits each): removing either one raises the
correct token's loss from 1.0 to about 5.3 bits.
The program claims (edges not measured here; the checker's edge cuts test them) that layer 1 is the
previous-token step, so that later keys say which token came before them; that layer 2 (key 206,
output 735) is the induction step: its key reads layer 1's write, its value carries the token to
copy, and its output writes it; and that layer 3 (output 806) writes it again after its value reads
layer 2. Layer 0's subcomponents also cost a lot here; these measurements do not show their role, so
the program only lets their write reach the later layers.
Left to their averages: the MLP subcomponents, although some cost a lot when removed (layer 1 c_fc
2103: 9.6 bits, layer 0 c_fc 53: 7.3, layer 0 down_proj 3257: 2.3, layer 1 down_proj 1217: 2.2, layer
3 down_proj 3532: 0.6). The claim is that what they contribute here barely depends on the prompt.
"""
from mech import node, edges, PD, embed, logits

early = node(
    PD.vpd[0].q_proj[28, 378],             # 0.39, 0.09
    PD.vpd[0].k_proj[29, 309, 465, 299],   # 0.62, 0.58, 0.08, 0.07
    PD.vpd[0].v_proj[82, 948, 49],         # 0.25, 0.22, 0.21
    PD.vpd[0].o_proj[389, 329, 437],       # 1.55, 0.10, 0.06
)
prev = node(
    PD.vpd[1].q_proj[316],                 # 3.77
    PD.vpd[1].k_proj[329],                 # 3.84
    PD.vpd[1].v_proj[228, 346, 72, 1000, 919, 428],   # 0.08 .. 0.06 each
    PD.vpd[1].o_proj[311, 340, 37, 630],   # 0.37, 0.10, 0.07, 0.07
)
induction = node(
    PD.vpd[2].q_proj[335, 207, 279],       # 0.28 (removing it raises the correct token), 0.08, 0.06
    PD.vpd[2].k_proj[206, 1, 224, 286, 373],   # 0.45, then 0.08 .. 0.05
    PD.vpd[2].v_proj[559],                 # 0.07
    PD.vpd[2].o_proj[735],                 # 0.69
)
copy = node(
    PD.vpd[3].q_proj[334],                 # 0.24
    PD.vpd[3].k_proj[145, 261],            # 0.38, 0.06
    PD.vpd[3].v_proj[677],                 # 0.07
    PD.vpd[3].o_proj[806],                 # 0.35
)

edges(
    embed >> early,
    early >> prev,                 # layer 0's write reaches every read of the later layers
    early >> induction,
    early >> copy,
    embed >> prev,                 # the previous token's identity, moved forward one position
    prev >> induction.key,         # key at j+1: "the token before me was A"
    embed >> induction.query,      # query at t: "I am A"
    embed >> induction.value,      # value at j+1: B, the token to copy
    induction >> copy.value,
    induction >> logits,           # B, written directly
    copy >> logits,                # B again
)

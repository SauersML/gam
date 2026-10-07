"""Induction on vpd4l: when the current token appeared earlier in the context, predict the token that
followed it there ("A B ... A -> B").

Three heads form the circuit (the repo's induction contracts, mpd_library_contracts_2951.rs):
a previous-token head writes, at every position, the token before it; the induction head's key reads
that write, so a query for the current token A matches the position right after the earlier A; the
induction head copies the token there (B) into the stream; a later copy head reads the induction head
through its value and writes B again, boosting it in the logits.
"""
from mech import node, edges, L, embed, logits

prev = node(L[1].head[1])       # previous-token head: attends from t to t-1, writes x[t-1]
induction = node(L[2].head[4])  # attends from the current A to the position after the earlier A
copy = node(L[3].head[5])       # reads the induction head's write through its value, writes B again

edges(
    embed >> prev.query,        # position only (rotary): the query needs little content
    embed >> prev.key,
    embed >> prev.value,        # the token it moves forward one position
    prev >> induction.key,      # key at j+1 = "the token before me was A"
    embed >> induction.query,   # query at t = "I am A"
    embed >> induction.value,   # value at j+1 = B, the token to copy
    induction >> copy.value,
    induction >> logits,        # the induction head writes B directly
    copy >> logits,             # and the copy head writes it again
)

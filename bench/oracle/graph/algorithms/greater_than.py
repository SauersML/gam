"""Greater-than: "from 1816 to 18" -> a year after the start within its century: the end year's last two
digits exceed the start year's. The answer here is the smallest such year."""
from mech import bind, claim


def start(tokens):
    # the start year, once its four digits are written
    out, digits = [], ""
    for tok in tokens:
        if tok.strip().isdigit() and len(digits) < 4:
            digits += tok.strip()
        out.append(int(digits) if len(digits) == 4 else None)
    return out


def answer(tokens, start):
    # the year after the start; once its century is written, the last two digits
    out = []
    for t, y in enumerate(start):
        if not t or start[t - 1] is None:  # the start is not complete before t
            out.append(None)
        elif tokens[t].strip() == str(y // 100):
            out.append("%02d" % (y % 100 + 1))
        else:
            out.append(" " + str(y + 1))
    return out

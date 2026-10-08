"""Greater-than: "from 1816 to 18" -> a year after the start within its century: the end year's last two
digits exceed the start year's. Any such year is right."""
from mech import align, claim


def start(tokens):
    # the start year: the first four-digit number, once written
    out = []
    for t in range(len(tokens)):
        digits = "".join(c for c in "".join(tokens[: t + 1]) if c.isdigit())
        out.append(int(digits[:4]) if len(digits) >= 4 else None)
    return out


def rest(full, text):
    # what is left of `full` after the longest beginning of it that the text ends with
    return full[max(n for n in range(len(full)) if text.endswith(full[:n])):]


def answer(tokens, start):
    # every later year of the start's century, or what is left of each once its first digits are written
    out = []
    for t, y in enumerate(start):
        text = "".join(tokens[: t + 1])
        if y is None or text.rstrip().endswith(str(y)):
            out.append(None)
        else:
            out.append(sorted({rest(" " + str(later), text) for later in range(y + 1, y // 100 * 100 + 100)}) or None)
    return out

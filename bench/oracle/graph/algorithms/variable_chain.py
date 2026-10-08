"""Variable binding: a value assigned and copied along a chain of variables ("a = 21, p = a, c = p");
printing the last variable prints the value."""
from mech import bind, claim


def values(tokens):
    # each variable's value, following the assignments in order
    out = []
    for t in range(len(tokens)):
        env = {}
        for line in "".join(tokens[: t + 1]).split("\n"):
            if " = " in line:
                name, value = line.split(" = ", 1)
                env[name.strip()] = value.strip() if value.strip().isdigit() else env.get(value.strip())
        out.append(env)
    return out


def rest(full, text):
    # what is left of `full` after the longest beginning of it that the text ends with
    return full[max(n for n in range(len(full)) if text.endswith(full[:n])):]


def answer(tokens, values):
    # the value of the printed variable, after "# prints"
    out = []
    for t, env in enumerate(values):
        text = "".join(tokens[: t + 1])
        name = text.rsplit("print(", 1)[1].split(")")[0] if "print(" in text else None
        value = env.get(name) if name else None
        out.append(None if value is None or "prints" not in text else rest(" " + value, text))
    return out

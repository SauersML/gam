"""Multiple choice: the answer is the letter of the option that answers the question. The table of right
answers is what the model recalls."""
from mech import align, claim

RIGHT = {
    'How many days are in a week?': 'seven', 'How many legs does a dog have?': 'four',
    'What color is grass?': 'green', 'What color is the sky on a clear day?': 'blue',
    'What do bees make?': 'honey', 'What do cows drink?': 'water',
    'What do we use to write on paper?': 'pen', 'What is 2 plus 2?': '4',
    'What is frozen water called?': 'ice', 'What is the capital of France?': 'Paris',
    'What is the opposite of hot?': 'cold', 'Which animal is the largest?': 'whale',
    'Which animal says meow?': 'cat', 'Which fruit is yellow and long?': 'banana',
    'Which planet do we live on?': 'Earth', 'Which season is the coldest?': 'winter',
}


def right(tokens):
    # the right answer to the question asked
    out = []
    for t in range(len(tokens)):
        text = "".join(tokens[: t + 1])
        question = text.split("Question: ", 1)[1].split("\n")[0] if "Question: " in text else None
        out.append(RIGHT.get(question))
    return out


def options(tokens):
    # the lettered options written so far: {option: letter}
    out = []
    for t in range(len(tokens)):
        lines = "".join(tokens[: t + 1]).split("\n")
        out.append({line[3:].strip(): line[0] for line in lines[:-1] if len(line) > 3 and line[1:3] == ". "})
    return out


def answer(tokens, right, options):
    # the letter of the right option, after "Answer:"
    out = []
    for t, (r, opts) in enumerate(zip(right, options)):
        text = "".join(tokens[: t + 1])
        out.append(" " + opts[r] if r in opts and text.endswith("Answer:") else None)
    return out

"""Behavior families for the graph oracle (MPD #2951): templates with exact answers.

A family returns variants; a variant is a list of items. An item is a prompt prefix, its answer, and a
counterfactual (a minimal edit of the prefix that changes the answer). An item without a counterfactual
carries a `tmpl` key, and build.py pairs it with another item of the same variant and template whose
prefix and answer have the same token lengths (fact families: swap the entity).

Every family is generated for every model; build.py measures the model and keeps a behavior only when the
answer is the model's top-1 token on most targets. Generators receive `tok`, which answers tokenizer
questions (`tok.single(" word")`: one token), so word lists can be filtered per model.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

SVA_DIR = Path.home() / "mpd-data/circuits/sva"


@dataclass
class Item:
    prefix: str
    answer: str
    cf_prefix: str | None = None
    cf_answer: str | None = None
    tmpl: int = 0
    accept: list[str] | None = None  # other correct answers (greater-than, rhyme); scoring uses their first token
    cf_accept: list[str] | None = None


@dataclass
class Variant:
    name: str
    description: str
    items: list[Item] = field(default_factory=list)


FAMILIES: dict[str, callable] = {}


def family(fn):
    FAMILIES[fn.__name__] = fn
    return fn


def words(tok, ws, n=1):
    """Keep the words that are `n` tokens with a leading space."""
    return [w for w in ws if tok.count(" " + w) == n]


NAMES_F = "Mary Anna Sarah Emma Lisa Laura Kate Julia Rachel Linda Susan Alice Grace Helen Emily Rose Amy Jane Claire Diana Nancy Karen Sophie Olivia".split()
NAMES_M = "John David Michael James Robert Tom Peter Paul Mark Daniel Kevin George Steven Richard Brian Eric Frank Jack Henry Adam Ryan Chris Scott Andrew".split()
PLACES = "store park school office market station hospital library beach restaurant garden museum".split()
OBJECTS = "drink book ball letter key gift ring pen bag cake flower hat".split()
NOUNS = ("apple chair river lamp window garden pencil mountain bottle candle doctor table spoon forest kitten bridge "
         "rocket guitar island blanket camera dragon ladder mirror pillow tunnel violin wallet basket button cookie "
         "engine feather hammer jacket ocean planet rabbit saddle tiger tomato bucket carpet desert dolphin anchor "
         "barrel castle cherry circus coffee cotton crystal diamond falcon finger garlic glove helmet honey jungle "
         "lemon magnet marble needle orange parrot pepper pirate puzzle rainbow ribbon robot salmon shadow silver "
         "spider sugar summer temple thunder ticket tower velvet wagon whale winter wizard zebra").split()


@family
def ioi(tok, rng):
    """Indirect object identification: the name that appeared once is the answer."""
    names = words(tok, NAMES_F + NAMES_M)
    templates = [
        ("store", "When {A} and {B} went to the {place}, {S} gave a {obj} to"),
        ("argument", "Then, {A} and {B} had a long argument. Afterwards {S} said to"),
        ("work", "{A} and {B} were working at the {place}. {S} decided to give a {obj} to"),
    ]
    out = []
    for vname, t in templates:
        v = Variant(f"{vname}", f"IOI ({vname} template): after two names, the second mention of one name (the subject) is followed by the other name (the indirect object).")
        for _ in range(96):
            a, b = rng.sample(names, 2)
            s = rng.choice([a, b])
            io = b if s == a else a
            kw = dict(A=a, B=b, place=rng.choice(PLACES), obj=rng.choice(OBJECTS))
            v.items.append(Item(t.format(S=s, **kw), " " + io, t.format(S=io, **kw), " " + s))
        out.append(v)
    return out


INDUCTION_WORDS = NOUNS + ("blue green quick slow happy angry silent bright golden purple frozen hidden wooden ancient "
                           "gentle bitter").split()


@family
def induction_random(tok, rng):
    """Induction on a repeated list of random words: after the repeat starts, continue the copy."""
    ws = words(tok, INDUCTION_WORDS)
    out = []
    for n in (8, 16):
        v = Variant(f"words{n}", f"Induction: a list of {n} random words is repeated; at a point in the repeat the next word is the one that followed the same word in the first copy.")
        for _ in range(96):
            seq = rng.sample(ws, n + 1)
            first, spare = seq[:n], seq[n]
            k = rng.randrange(2, n - 1)
            alt = list(first)
            alt[k] = spare
            v.items.append(Item(" " + " ".join(first) + " ." + " " + " ".join(first[:k]), " " + first[k],
                                " " + " ".join(alt) + " ." + " " + " ".join(alt[:k]), " " + spare))
        out.append(v)
    return out


SURNAMES = ("Zalinski Montague Okafor Brennan Castellano Whitfield Abernathy Kowalczyk Delacroix Featherstone Hargreaves "
            "Lindqvist Nakamura Pemberton Quarrington Rasmussen Thornbury Vasquez Wainwright Yarborough Fitzgerald "
            "Galloway Holloway Kingsley Mendoza Ferreira Oyelaran Petrakis Szymanski Tremblay").split()


@family
def name_repeat(tok, rng):
    """Copy a two-part name: a first name seen before is followed by the same surname."""
    firsts = words(tok, NAMES_F + NAMES_M)
    cities = words(tok, "Boston Paris London Chicago Berlin Tokyo Madrid Denver Austin Dublin".split())
    out = []
    v1 = Variant("record", "Name repetition in a record: a first name that appeared earlier with a surname is followed by that surname.")
    v2 = Variant("story", "Name repetition in prose: a person introduced by full name is later mentioned by first name and the surname follows.")
    while len(v1.items) < 128:
        f = rng.choice(firsts)
        s1, s2 = rng.sample(SURNAMES, 2)
        if tok.count(" " + s1) != tok.count(" " + s2):
            continue
        age, city = rng.randrange(21, 79), rng.choice(cities)
        t = "Name: {F} {S}\nAge: {age}\nCity: {city}\n\nName: {F}"
        v1.items.append(Item(t.format(F=f, S=s1, age=age, city=city), " " + s1, t.format(F=f, S=s2, age=age, city=city), " " + s2))
        t2 = "Our new neighbor is {F} {S}. Everyone on the street likes {F}"
        v2.items.append(Item(t2.format(F=f, S=s1), " " + s1, t2.format(F=f, S=s2), " " + s2))
    out += [v1, v2]
    return out


@family
def sva(tok, rng):
    """Subject-verb agreement (Marks et al. pairs): the verb agrees in number with the subject."""
    out = []
    for name, desc in [("simple", "a determiner and noun"), ("nounpp", "a noun followed by a prepositional phrase with a distractor noun"),
                       ("rc", "a noun followed by a relative clause with a distractor noun"), ("within_rc", "the verb inside a relative clause agreeing with that clause's subject")]:
        v = Variant(name, f"Subject-verb agreement after {desc}; the counterfactual flips the subject's number.")
        rows = [json.loads(l) for f in ("train", "test") for l in (SVA_DIR / f"{name}_{f}.json").read_text().splitlines()]
        rng.shuffle(rows)
        seen = set()
        for r in rows:
            if r["clean_prefix"] + r["clean_answer"] in seen:
                continue
            seen.add(r["clean_prefix"] + r["clean_answer"])
            v.items.append(Item(r["clean_prefix"], r["clean_answer"], r["patch_prefix"], r["patch_answer"]))
            if len(v.items) >= 128:
                break
        out.append(v)
    return out


@family
def greater_than(tok, rng):
    """Greater-than on years: the end year of a span must be later than the start year."""
    out = []
    for vname, t in [("war", "The war lasted from the year {C}{Y:02d} to the year {C}"),
                     ("contract", "The contract was valid from {C}{Y:02d} until {C}"),
                     ("reign", "The king ruled from {C}{Y:02d} to {C}")]:
        v = Variant(vname, "Greater-than: the end year starts with the same century, so its last two digits must exceed the start year's.")
        for _ in range(96):
            c = rng.choice([15, 16, 17, 18])
            y, y2 = rng.sample(range(2, 90), 2)
            ok = [f"{z:02d}" for z in range(y + 1, 100)]
            ok2 = [f"{z:02d}" for z in range(y2 + 1, 100)]
            v.items.append(Item(t.format(C=c, Y=y), rng.choice(ok), t.format(C=c, Y=y2), rng.choice(ok2), accept=ok, cf_accept=ok2))
        out.append(v)
    return out


COUNTRIES = [  # country, capital, language, continent, demonym adjective
    ("France", "Paris", "French", "Europe", "French"), ("Germany", "Berlin", "German", "Europe", "German"),
    ("Italy", "Rome", "Italian", "Europe", "Italian"), ("Spain", "Madrid", "Spanish", "Europe", "Spanish"),
    ("Japan", "Tokyo", "Japanese", "Asia", "Japanese"), ("China", "Beijing", "Chinese", "Asia", "Chinese"),
    ("Russia", "Moscow", "Russian", "Europe", "Russian"), ("Egypt", "Cairo", "Arabic", "Africa", "Egyptian"),
    ("Greece", "Athens", "Greek", "Europe", "Greek"), ("Portugal", "Lisbon", "Portuguese", "Europe", "Portuguese"),
    ("Poland", "Warsaw", "Polish", "Europe", "Polish"), ("Austria", "Vienna", "German", "Europe", "Austrian"),
    ("Hungary", "Budapest", "Hungarian", "Europe", "Hungarian"), ("Sweden", "Stockholm", "Swedish", "Europe", "Swedish"),
    ("Norway", "Oslo", "Norwegian", "Europe", "Norwegian"), ("Finland", "Helsinki", "Finnish", "Europe", "Finnish"),
    ("Denmark", "Copenhagen", "Danish", "Europe", "Danish"), ("Ireland", "Dublin", "English", "Europe", "Irish"),
    ("Turkey", "Ankara", "Turkish", "Asia", "Turkish"), ("Iran", "Tehran", "Persian", "Asia", "Iranian"),
    ("Iraq", "Baghdad", "Arabic", "Asia", "Iraqi"), ("India", "Delhi", "Hindi", "Asia", "Indian"),
    ("Thailand", "Bangkok", "Thai", "Asia", "Thai"), ("Vietnam", "Hanoi", "Vietnamese", "Asia", "Vietnamese"),
    ("Indonesia", "Jakarta", "Indonesian", "Asia", "Indonesian"), ("Korea", "Seoul", "Korean", "Asia", "Korean"),
    ("Kenya", "Nairobi", "Swahili", "Africa", "Kenyan"), ("Nigeria", "Abuja", "English", "Africa", "Nigerian"),
    ("Ethiopia", "Addis", "Amharic", "Africa", "Ethiopian"), ("Morocco", "Rabat", "Arabic", "Africa", "Moroccan"),
    ("Peru", "Lima", "Spanish", "South America", "Peruvian"), ("Chile", "Santiago", "Spanish", "South America", "Chilean"),
    ("Argentina", "Buenos", "Spanish", "South America", "Argentine"), ("Brazil", "Brasilia", "Portuguese", "South America", "Brazilian"),
    ("Colombia", "Bogota", "Spanish", "South America", "Colombian"), ("Venezuela", "Caracas", "Spanish", "South America", "Venezuelan"),
    ("Mexico", "Mexico", "Spanish", "North America", "Mexican"), ("Canada", "Ottawa", "English", "North America", "Canadian"),
    ("Cuba", "Havana", "Spanish", "North America", "Cuban"), ("Australia", "Canberra", "English", "Australia", "Australian"),
    ("Belgium", "Brussels", "Dutch", "Europe", "Belgian"), ("Netherlands", "Amsterdam", "Dutch", "Europe", "Dutch"),
    ("Switzerland", "Bern", "German", "Europe", "Swiss"), ("Ukraine", "Kyiv", "Ukrainian", "Europe", "Ukrainian"),
    ("Romania", "Bucharest", "Romanian", "Europe", "Romanian"), ("Bulgaria", "Sofia", "Bulgarian", "Europe", "Bulgarian"),
    ("Serbia", "Belgrade", "Serbian", "Europe", "Serbian"), ("Croatia", "Zagreb", "Croatian", "Europe", "Croatian"),
    ("Pakistan", "Islamabad", "Urdu", "Asia", "Pakistani"), ("Afghanistan", "Kabul", "Pashto", "Asia", "Afghan"),
    ("Syria", "Damascus", "Arabic", "Asia", "Syrian"), ("Lebanon", "Beirut", "Arabic", "Asia", "Lebanese"),
    ("Israel", "Jerusalem", "Hebrew", "Asia", "Israeli"), ("Philippines", "Manila", "Filipino", "Asia", "Filipino"),
    ("Malaysia", "Kuala", "Malay", "Asia", "Malaysian"), ("Mongolia", "Ulaanbaatar", "Mongolian", "Asia", "Mongolian"),
    ("Nepal", "Kathmandu", "Nepali", "Asia", "Nepalese"), ("Ghana", "Accra", "English", "Africa", "Ghanaian"),
    ("Senegal", "Dakar", "French", "Africa", "Senegalese"), ("Uganda", "Kampala", "English", "Africa", "Ugandan"),
    ("Tanzania", "Dodoma", "Swahili", "Africa", "Tanzanian"), ("Iceland", "Reykjavik", "Icelandic", "Europe", "Icelandic"),
    ("Scotland", "Edinburgh", "English", "Europe", "Scottish"), ("Wales", "Cardiff", "Welsh", "Europe", "Welsh"),
    ("Czechia", "Prague", "Czech", "Europe", "Czech"), ("Slovakia", "Bratislava", "Slovak", "Europe", "Slovak"),
    ("Ecuador", "Quito", "Spanish", "South America", "Ecuadorian"), ("Bolivia", "Sucre", "Spanish", "South America", "Bolivian"),
    ("Uruguay", "Montevideo", "Spanish", "South America", "Uruguayan"), ("Jamaica", "Kingston", "English", "North America", "Jamaican"),
    ("Libya", "Tripoli", "Arabic", "Africa", "Libyan"), ("Sudan", "Khartoum", "Arabic", "Africa", "Sudanese"),
]


MIN_PROMPTS = 64  # the suite's floor of prompts per behavior


def fact_variants(rows, templates, desc, name=None):
    """One variant per template when the rows fill one; otherwise one variant mixing all templates. Items carry
    `tmpl` so build.py pairs entities only within a template and with equal token lengths."""
    if len(rows) >= MIN_PROMPTS or len(templates) == 1:
        groups = [(vname, [(i, t)]) for i, (vname, t) in enumerate(templates)]
    else:
        groups = [(name or "mixed", list(enumerate(t for _, t in templates)))]
    out = []
    for vname, ts in groups:
        v = Variant(vname, desc)
        for i, t in ts:
            for subj, ans in rows:
                v.items.append(Item(t.format(X=subj), " " + ans, tmpl=i))
        out.append(v)
    return out


@family
def capital(tok, rng):
    """Country capitals."""
    rows = [(c[0], c[1]) for c in COUNTRIES]
    return fact_variants(rows, [("plain", "The capital of {X} is"), ("qa", "Q: What is the capital of {X}?\nA:"),
                                ("city", "{X}'s capital city is")], "Factual recall: the capital city of a named country.")


@family
def country_language(tok, rng):
    """The language spoken in a country."""
    rows = [(c[0], c[2]) for c in COUNTRIES]
    return fact_variants(rows, [("speak", "In {X}, most people speak"), ("official", "The official language of {X} is")],
                         "Factual recall: the main language of a named country.")


@family
def country_continent(tok, rng):
    """The continent of a country."""
    rows = [(c[0], c[3]) for c in COUNTRIES]
    return fact_variants(rows, [("located", "{X} is a country located in"), ("continent", "The continent where {X} lies is")],
                         "Factual recall: the continent of a named country.")


@family
def demonym(tok, rng):
    """The adjective for a person from a country."""
    rows = [(c[0], c[4]) for c in COUNTRIES]
    return fact_variants(rows, [("born", "She was born and raised in {X}, so she is"), ("citizen", "A citizen of {X} is called")],
                         "Factual recall: the nationality adjective of a named country.")


LANDMARKS = [("the Eiffel Tower", "Paris"), ("the Colosseum", "Rome"), ("Big Ben", "London"), ("the Kremlin", "Moscow"),
             ("the Statue of Liberty", "New"), ("the Brandenburg Gate", "Berlin"), ("the Acropolis", "Athens"),
             ("the Louvre", "Paris"), ("the Sagrada Familia", "Barcelona"), ("the Golden Gate Bridge", "San"),
             ("the Space Needle", "Seattle"), ("the Forbidden City", "Beijing"), ("the Taj Mahal", "Agra"),
             ("the Vatican", "Rome"), ("Buckingham Palace", "London"), ("the Opera House", "Sydney"),
             ("the Hollywood sign", "Los"), ("the Gateway Arch", "St"), ("the Tower of London", "London"),
             ("the Burj Khalifa", "Dubai"), ("Times Square", "New"), ("the Alhambra", "Granada"),
             ("the Little Mermaid statue", "Copenhagen"), ("Red Square", "Moscow"), ("the Parthenon", "Athens"),
             ("the Pantheon", "Rome"), ("the Trevi Fountain", "Rome"), ("Notre Dame Cathedral", "Paris"),
             ("the Prado Museum", "Madrid"), ("the Rijksmuseum", "Amsterdam"), ("the Brooklyn Bridge", "New"),
             ("the White House", "Washington"), ("the Hagia Sophia", "Istanbul"), ("the Blue Mosque", "Istanbul"),
             ("the Uffizi Gallery", "Florence"), ("the Leaning Tower", "Pisa"), ("Tower Bridge", "London"),
             ("the Arc de Triomphe", "Paris"), ("the CN Tower", "Toronto"), ("Wrigley Field", "Chicago"),
             ("Charles Bridge", "Prague"), ("the Petronas Towers", "Kuala"), ("Table Mountain", "Cape"),
             ("the Christ the Redeemer statue", "Rio"), ("the Bund", "Shanghai"), ("Tokyo Tower", "Tokyo"),
             ("the Grand Bazaar", "Istanbul"), ("the Atomium", "Brussels"), ("Schonbrunn Palace", "Vienna"),
             ("the Sydney Harbour Bridge", "Sydney"), ("the Alamo", "San"), ("the Liberty Bell", "Philadelphia"),
             ("the Hermitage Museum", "Saint"), ("Edinburgh Castle", "Edinburgh"), ("the Duomo", "Florence"),
             ("Westminster Abbey", "London"), ("the Smithsonian", "Washington"), ("the Empire State Building", "New"),
             ("the Rialto Bridge", "Venice"), ("St Mark's Basilica", "Venice"), ("the Mona Lisa", "Paris"),
             ("the Reichstag", "Berlin"), ("the Sistine Chapel", "Rome"), ("the Gherkin", "London")]


@family
def landmark_city(tok, rng):
    """The city of a landmark."""
    out = []
    for i, (vname, t) in enumerate([("located", "{X} is located in the city of"), ("visit", "To see {X}, tourists travel to")]):
        v = Variant(vname, "Factual recall: the city where a named landmark stands.")
        for subj, ans in LANDMARKS:
            v.items.append(Item(t.format(X=subj[0].upper() + subj[1:] if t.startswith("{X}") else subj), " " + ans, tmpl=i))
        out.append(v)
    return out


@family
def pronoun_gender(tok, rng):
    """A pronoun referring to a named person matches the name's usual gender."""
    f, m = words(tok, NAMES_F), words(tok, NAMES_M)
    out = []
    for vname, t in [("because", "{N} went to the {place} because"), ("said", "After {N} finished the work at the {place},"),
                     ("thinks", "{N} looked tired at the {place}, so I asked if")]:
        v = Variant(vname, "Gendered pronoun: the next word is the pronoun for the named person; the counterfactual swaps the name's gender.")
        for _ in range(200):
            a, b = rng.choice(f), rng.choice(m)
            p = rng.choice(PLACES)
            if rng.random() < 0.5:
                v.items.append(Item(t.format(N=a, place=p), " she", t.format(N=b, place=p), " he"))
            else:
                v.items.append(Item(t.format(N=b, place=p), " he", t.format(N=a, place=p), " she"))
        out.append(v)
    return out


@family
def bracket_close(tok, rng):
    """Close the innermost open bracket."""
    vs = list("abcdxyzmn")
    out = []
    v1 = Variant("paren", "Bracket closing in arithmetic: after a nested parenthesized expression the next token closes the open parentheses.")
    v2 = Variant("list", "Bracket closing in nested lists: after the last element the next token closes the open lists.")
    v3 = Variant("call", "Bracket closing in function calls: after the last argument the next token closes the open calls.")
    for _ in range(160):
        a, b, c = rng.sample(vs, 3)
        op1, op2 = rng.choice("+-*"), rng.choice("+-*")
        # counterfactual: the inner group closed earlier, so one parenthesis remains open
        # counterfactual: a minus sign replaces the inner opening bracket, so one bracket fewer is open
        v1.items.append(Item(f"y = ({a} {op1} ({b} {op2} {c}", "))", f"y = ({a} {op1} -{b} {op2} {c}", ")"))
        n = [rng.randrange(1, 10) for _ in range(4)]
        v2.items.append(Item(f"x = [{n[0]}, [{n[1]}, [{n[2]}, {n[3]}", "]]]", f"x = [{n[0]}, [{n[1]}, -{n[2]}, {n[3]}", "]]"))
        f1, f2 = rng.sample(["max", "min", "abs", "len", "sum", "int", "str", "round"], 2)
        v3.items.append(Item(f"value = {f1}({f2}({a}, {b}", "))", f"value = {f1}({f2}({a}), {b}", ")"))
    return [v1, v2, v3]


@family
def quote_close(tok, rng):
    """Close an open quotation."""
    subj = words(tok, NAMES_F + NAMES_M)
    phrases = ["I will be there soon", "this is not what I ordered", "the train leaves at noon", "we should go home",
               "please close the door", "the answer is simple", "nobody told me", "it is going to rain",
               "I love this song", "the meeting is cancelled", "we won the game", "dinner is ready"]
    v = Variant("said", "Quote closing: after a quoted sentence ends with a period, the next token closes the quotation; the counterfactual opens a parenthesis instead.")
    for _ in range(96):
        n, p = rng.choice(subj), rng.choice(phrases)
        # counterfactual: a parenthesis opens the aside instead of a quotation mark
        v.items.append(Item(f'{n} looked up and said, "{p.capitalize()}.', '"', f'{n} looked up and said, ({p.capitalize()}.', ")"))
    return [v]


@family
def code_variable(tok, rng):
    """Python: reuse the variable that the code defined."""
    names = words(tok, "item value total count result name entry record score price node user word line".split())
    out = []
    v1 = Variant("loop", "Python variable reuse: in a for loop body, the loop variable is printed.")
    v2 = Variant("return", "Python variable reuse: a function returns the local variable it just computed.")
    v3 = Variant("argument", "Python variable reuse: a function body uses its only argument.")
    for _ in range(200):
        a, b = rng.sample(names, 2)
        seq = rng.choice(["data", "rows", "items", "values", "entries"])
        v1.items.append(Item(f"for {a} in {seq}:\n    print(", a, f"for {b} in {seq}:\n    print(", b))
        f = rng.choice(["compute", "combine", "merge", "update", "measure"])
        v2.items.append(Item(f"def {f}(x, y):\n    {a} = x + y\n    return", " " + a, f"def {f}(x, y):\n    {b} = x + y\n    return", " " + b))
        v3.items.append(Item(f"def {f}({a}):\n    if {a} is None:\n        return None\n    return len(", a,
                             f"def {f}({b}):\n    if {b} is None:\n        return None\n    return len(", b))
    return [v1, v2, v3]


@family
def docstring_args(tok, rng):
    """Python: the docstring's Args section lists the signature's arguments in order."""
    names = words(tok, "alpha beta gamma delta width height depth start stop step size count limit offset scale label".split())
    v = Variant("args", "Docstring argument recall: after two documented arguments, the next documented name is the third argument of the signature.")
    for _ in range(96):
        a, b, c, d = rng.sample(names, 4)
        t = 'def run({a}, {b}, {c}):\n    """Run the job.\n\n    Args:\n        {a}: the first setting.\n        {b}: the second setting.\n       '
        v.items.append(Item(t.format(a=a, b=b, c=c), " " + c, t.format(a=a, b=b, c=d), " " + d))
    return [v]


PLURALS_REG = "cat dog car book tree house bird chair pen cup shoe star boat lamp door key bell coin hat bag apple flower river ring desk girl boy friend road farm".split()
PLURALS_IRR = [("mouse", "mice"), ("child", "children"), ("man", "men"), ("woman", "women"), ("foot", "feet"), ("tooth", "teeth"),
               ("goose", "geese"), ("person", "people"), ("knife", "knives"), ("wolf", "wolves"), ("leaf", "leaves"),
               ("wife", "wives"), ("life", "lives"), ("half", "halves"), ("ox", "oxen"), ("cactus", "cacti"), ("box", "boxes"),
               ("bus", "buses"), ("city", "cities"), ("baby", "babies"), ("party", "parties"), ("story", "stories"),
               ("glass", "glasses"), ("church", "churches"), ("dish", "dishes"), ("fox", "foxes"), ("lady", "ladies"),
               ("puppy", "puppies"), ("hero", "heroes"), ("potato", "potatoes"), ("tomato", "tomatoes"), ("loaf", "loaves")]


@family
def plural(tok, rng):
    """Plural nouns after a number."""
    reg = [(w, w + "s") for w in PLURALS_REG]
    ts = [("have", "I have one {X} and you have two"), ("list", "one {X}, two"), ("there", "There is one {X} here and three")]
    return (fact_variants(reg, ts, "Plural of a regular noun after a count above one.", "regular") +
            fact_variants(PLURALS_IRR, ts, "Plural of an irregular or spelling-changing noun after a count above one.", "irregular"))


PAST_REG = "walk talk play jump cook clean paint watch call open help work visit wash climb dance laugh listen pull push start finish travel kick".split()
PAST_IRR = [("go", "went"), ("eat", "ate"), ("see", "saw"), ("run", "ran"), ("write", "wrote"), ("drink", "drank"), ("swim", "swam"),
            ("sing", "sang"), ("drive", "drove"), ("buy", "bought"), ("bring", "brought"), ("think", "thought"), ("teach", "taught"),
            ("catch", "caught"), ("fly", "flew"), ("speak", "spoke"), ("break", "broke"), ("take", "took"), ("give", "gave"),
            ("make", "made"), ("sleep", "slept"), ("leave", "left"), ("feel", "felt"), ("sit", "sat"), ("stand", "stood"),
            ("win", "won"), ("lose", "lost"), ("find", "found"), ("tell", "told"), ("sell", "sold"), ("ride", "rode")]


@family
def past_tense(tok, rng):
    """Past tense after a habitual present."""
    reg = [(w, w + "ed") for w in PAST_REG]
    ts = [("yesterday", "Every day I {X}. Yesterday I"), ("last_week", "Today they {X}. Last week they"),
          ("last_night", "I usually {X} in the morning, but last night I")]
    return (fact_variants(reg, ts, "Past tense of a regular verb.", "regular") + fact_variants(PAST_IRR, ts, "Past tense of an irregular verb.", "irregular"))


ANTONYMS = [("hot", "cold"), ("big", "small"), ("fast", "slow"), ("happy", "sad"), ("up", "down"), ("light", "dark"),
            ("old", "young"), ("rich", "poor"), ("hard", "soft"), ("high", "low"), ("long", "short"), ("open", "closed"),
            ("early", "late"), ("full", "empty"), ("wet", "dry"), ("strong", "weak"), ("clean", "dirty"), ("heavy", "light"),
            ("thick", "thin"), ("loud", "quiet"), ("true", "false"), ("good", "bad"), ("love", "hate"), ("win", "lose"),
            ("buy", "sell"), ("push", "pull"), ("give", "take"), ("inside", "outside"), ("left", "right"), ("north", "south"),
            ("east", "west"), ("day", "night"), ("black", "white"), ("begin", "end"), ("first", "last"), ("cheap", "expensive"),
            ("safe", "dangerous"), ("easy", "difficult"), ("wide", "narrow"), ("deep", "shallow"), ("sweet", "sour"),
            ("smooth", "rough"), ("brave", "cowardly"), ("friend", "enemy"), ("question", "answer"), ("arrive", "leave"),
            ("accept", "reject"), ("increase", "decrease"), ("maximum", "minimum"), ("above", "below"), ("before", "after"),
            ("always", "never"), ("male", "female"), ("tall", "short"), ("near", "far"), ("alive", "dead"), ("asleep", "awake"),
            ("polite", "rude"), ("noisy", "quiet"), ("ancient", "modern"), ("upper", "lower"), ("major", "minor"),
            ("positive", "negative"), ("visible", "invisible"), ("possible", "impossible")]


@family
def antonym(tok, rng):
    """The opposite of a word."""
    return fact_variants(ANTONYMS, [("zero_shot", "The opposite of {X} is"),
                                    ("few_shot", "hot: cold\ntall: short\nyes: no\n{X}:")], "Antonym of a common word.")


@family
def number_successor(tok, rng):
    """Continue an arithmetic sequence of integers."""
    out = []
    for vname, step, desc in [("plus1", 1, "counting up by one"), ("plus2", 2, "counting up by two"), ("minus1", -1, "counting down by one"),
                              ("plus10", 10, "counting up by ten")]:
        v = Variant(vname, f"Number succession: a comma-separated sequence {desc}; the next number continues it.")
        while len(v.items) < 160:
            digits = rng.choice([2, 3])
            s, s2 = (rng.randrange(10 ** (digits - 1), 10 ** digits) for _ in range(2))
            seq = [s + step * i for i in range(5)]
            seq2 = [s2 + step * i for i in range(5)]
            if s == s2 or any(len(str(x)) != digits for x in seq + seq2):
                continue
            v.items.append(Item(", ".join(map(str, seq[:4])) + ",", " " + str(seq[4]), ", ".join(map(str, seq2[:4])) + ",", " " + str(seq2[4])))
        out.append(v)
    return out


NUMWORDS = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
ORDINALS = "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth".split()


LIST_TEMPLATES = ["{L},", "Here is the order: {L},", "She said them in order: {L},", "Sequence: {L},"]


def sequence_items(lst, rng, cyclic, after_templates=(), n=200):
    """Runs of 2-4 consecutive items in list templates, plus 'the one after X' templates; the counterfactual
    starts the run elsewhere in the same template."""
    items, L = [], len(lst)
    for _ in range(n):
        if after_templates and rng.random() < 0.3:
            t = rng.choice(after_templates)
            i, j = rng.sample(range(L if cyclic else L - 1), 2)
            items.append(Item(t.format(X=lst[i]), " " + lst[(i + 1) % L], t.format(X=lst[j]), " " + lst[(j + 1) % L]))
            continue
        k = rng.randrange(2, 5)
        t = rng.choice(LIST_TEMPLATES)
        span = L if cyclic else L - k
        i, j = rng.sample(range(span), 2)
        run = lambda a: ", ".join(lst[(a + q) % L] for q in range(k))
        items.append(Item(t.format(L=run(i)), " " + lst[(i + k) % L], t.format(L=run(j)), " " + lst[(j + k) % L]))
    return items


@family
def word_successor(tok, rng):
    """Continue a sequence of number words or ordinals."""
    return [Variant("cardinal", "Number-word succession: a run of number words; the next word continues the count.",
                    sequence_items(NUMWORDS, rng, False, ["The number after {X} is", "Count on from {X}: the next one is"])),
            Variant("ordinal", "Ordinal succession: a run of ordinal words; the next word continues the order.",
                    sequence_items(ORDINALS, rng, False, ["The one after the {X} is the"]))]


DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
MONTHS = "January February March April May June July August September October November December".split()


@family
def day_successor(tok, rng):
    """The next day of the week."""
    return [Variant("mixed", "Weekday succession: the day that follows in a run of weekdays, or the day after a named one.",
                    sequence_items(DAYS, rng, True, ["The day after {X} is", "If today is {X}, tomorrow is", "Every {X} is followed by"]))]


@family
def month_successor(tok, rng):
    """The next month."""
    return [Variant("mixed", "Month succession: the month that follows in a run of months, or the month after a named one.",
                    sequence_items(MONTHS, rng, True, ["The month after {X} is", "If this month is {X}, next month is", "{X} is followed by"]))]


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@family
def alphabet(tok, rng):
    """The next letter of the alphabet."""
    return [Variant("upper", "Alphabet succession: a run of capital letters; the next letter continues it.",
                    sequence_items(list(LETTERS), rng, False, ["The letter after {X} is"])),
            Variant("lower", "Alphabet succession: a run of lowercase letters; the next letter continues it.",
                    sequence_items(list(LETTERS.lower()), rng, False, ["The letter after {X} is"]))]


ACRO_WORDS = ("Federal National Central Global United Royal Digital Public Advanced Applied Bureau Council Office Agency Institute "
              "Research Science Energy Health Market Network Security Training Systems Planning Development Transport Water "
              "Environment Finance Education Language Medical Ocean Space Trade Union Kingdom Investment Labor Product").split()


@family
def acronym(tok, rng):
    """The acronym of a capitalized name, inside parentheses."""
    ws = words(tok, ACRO_WORDS)
    v = Variant("three_word", "Acronym: after a three-word capitalized name and an open parenthesis, the next tokens spell its initials.")
    for _ in range(128):
        a, b, c = rng.sample(ws, 3)
        b2 = rng.choice([w for w in ws if w[0] != b[0] and w not in (a, c)])
        v.items.append(Item(f"The {a} {b} {c} (", a[0] + b[0] + c[0], f"The {a} {b2} {c} (", a[0] + b2[0] + c[0]))
    return [v]


MC_FACTS = [("What color is the sky on a clear day?", "blue", ["green", "red", "purple"]),
            ("How many legs does a dog have?", "four", ["two", "six", "eight"]),
            ("What is the capital of France?", "Paris", ["London", "Berlin", "Madrid"]),
            ("Which animal says meow?", "cat", ["cow", "dog", "duck"]),
            ("What is frozen water called?", "ice", ["steam", "sand", "glass"]),
            ("Which planet do we live on?", "Earth", ["Mars", "Venus", "Jupiter"]),
            ("What do bees make?", "honey", ["milk", "silk", "wool"]),
            ("How many days are in a week?", "seven", ["five", "six", "ten"]),
            ("What color is grass?", "green", ["blue", "orange", "pink"]),
            ("Which season is the coldest?", "winter", ["summer", "spring", "autumn"]),
            ("What is the opposite of hot?", "cold", ["warm", "big", "fast"]),
            ("Which animal is the largest?", "whale", ["mouse", "cat", "frog"]),
            ("What do cows drink?", "water", ["milk", "oil", "juice"]),
            ("What is 2 plus 2?", "4", ["3", "5", "6"]),
            ("Which fruit is yellow and long?", "banana", ["apple", "grape", "cherry"]),
            ("What do we use to write on paper?", "pen", ["spoon", "fork", "shoe"])]


@family
def mc_letter(tok, rng):
    """Multiple-choice answering with a letter."""
    v = Variant("abcd", "Multiple-choice letter: a question with four lettered options; the answer is the letter of the correct option. The counterfactual moves the correct option to another letter.")
    for _ in range(128):
        q, right, wrong = rng.choice(MC_FACTS)
        opts = [right] + list(wrong)
        rng.shuffle(opts)
        i = opts.index(right)
        j = rng.choice([k for k in range(4) if k != i])
        alt = list(opts)
        alt[i], alt[j] = alt[j], alt[i]
        fmt = lambda o: f"Question: {q}\n" + "".join(f"{L}. {x}\n" for L, x in zip("ABCD", o)) + "Answer:"
        v.items.append(Item(fmt(opts), " " + "ABCD"[i], fmt(alt), " " + "ABCD"[j]))
    return [v]


@family
def arithmetic(tok, rng):
    """Single-digit arithmetic in a few-shot format."""
    out = []
    for vname, sym, fn in [("add", "+", lambda a, b: a + b), ("sub", "-", lambda a, b: a - b), ("mul", "*", lambda a, b: a * b)]:
        v = Variant(vname, f"Arithmetic: after two worked examples, the result of a {vname} problem on small numbers.")
        for _ in range(300):
            x1, y1, x2, y2 = (rng.randrange(2, 10) for _ in range(4))
            if vname == "sub":
                x1, y1, x2, y2 = max(x1, y1), min(x1, y1), max(x2, y2), min(x2, y2)
            shots = f"{x1} {sym} {y1} = {fn(x1, y1)}\n{x2} {sym} {y2} = {fn(x2, y2)}\n"
            a, b = rng.randrange(2, 10), rng.randrange(1, 10)
            if vname == "sub" and b > a:
                a, b = b, a
            b2 = rng.choice([x for x in range(1, 10) if x != b and (vname != "sub" or x <= a)])
            r, r2 = fn(a, b), fn(a, b2)
            if len(str(r)) != len(str(r2)):
                continue
            v.items.append(Item(shots + f"{a} {sym} {b} =", f" {r}", shots + f"{a} {sym} {b2} =", f" {r2}"))
        out.append(v)
    return out


OBJ_COLORS = [("banana", "yellow"), ("lemon", "yellow"), ("snow", "white"), ("milk", "white"), ("grass", "green"), ("leaf", "green"),
              ("sky", "blue"), ("ocean", "blue"), ("blood", "red"), ("tomato", "red"), ("strawberry", "red"), ("coal", "black"),
              ("crow", "black"), ("carrot", "orange"), ("pumpkin", "orange"), ("orange", "orange"), ("lime", "green"),
              ("cucumber", "green"), ("cherry", "red"), ("sun", "yellow"), ("chocolate", "brown"), ("coffee", "brown"),
              ("eggplant", "purple"), ("grape", "purple"), ("flamingo", "pink"), ("frog", "green"), ("cloud", "white"),
              ("tire", "black"), ("fire truck", "red"), ("polar bear", "white"), ("emerald", "green"), ("ruby", "red"),
              ("sapphire", "blue"), ("lavender", "purple"), ("pig", "pink"), ("bark", "brown")]


@family
def object_color(tok, rng):
    """The typical color of an object."""
    return fact_variants(OBJ_COLORS, [("color_of", "The color of a {X} is usually"), ("qa", "Q: What color is a {X}?\nA: It is")],
                         "Commonsense recall: the typical color of a named object.")


@family
def list_copy(tok, rng):
    """Copy a named position from a short list."""
    ws = words(tok, NOUNS)
    v1 = Variant("first", "List indexing: given a list of four words, the first word.")
    v2 = Variant("last", "List indexing: given a list of four words, the last word.")
    for _ in range(96):
        l = rng.sample(ws, 5)
        a = l[:4]
        b1 = [l[4]] + a[1:]
        b2 = a[:3] + [l[4]]
        f = lambda xs, pos: f"List: {', '.join(xs)}.\nThe {pos} word in the list is"
        v1.items.append(Item(f(a, "first"), " " + a[0], f(b1, "first"), " " + b1[0]))
        v2.items.append(Item(f(a, "last"), " " + a[3], f(b2, "last"), " " + b2[3]))
    return [v1, v2]


TRANSLATIONS = {
    "french": [("dog", "chien"), ("cat", "chat"), ("house", "maison"), ("water", "eau"), ("bread", "pain"), ("book", "livre"),
               ("car", "voiture"), ("sun", "soleil"), ("moon", "lune"), ("tree", "arbre"), ("apple", "pomme"), ("friend", "ami"),
               ("day", "jour"), ("night", "nuit"), ("red", "rouge"), ("green", "vert"), ("black", "noir"), ("white", "blanc"),
               ("man", "homme"), ("woman", "femme"), ("child", "enfant"), ("city", "ville"), ("school", "école"),
               ("cheese", "fromage"), ("milk", "lait"), ("fish", "poisson"), ("bird", "oiseau"), ("horse", "cheval"),
               ("door", "porte"), ("window", "fenêtre"), ("sea", "mer"), ("king", "roi"), ("time", "temps"), ("love", "amour")],
    "spanish": [("dog", "perro"), ("cat", "gato"), ("house", "casa"), ("water", "agua"), ("bread", "pan"), ("book", "libro"),
                ("car", "coche"), ("sun", "sol"), ("moon", "luna"), ("tree", "árbol"), ("apple", "manzana"), ("friend", "amigo"),
                ("day", "día"), ("night", "noche"), ("red", "rojo"), ("green", "verde"), ("black", "negro"), ("white", "blanco"),
                ("man", "hombre"), ("woman", "mujer"), ("child", "niño"), ("city", "ciudad"), ("school", "escuela"),
                ("cheese", "queso"), ("milk", "leche"), ("fish", "pescado"), ("bird", "pájaro"), ("horse", "caballo"),
                ("door", "puerta"), ("window", "ventana"), ("sea", "mar"), ("king", "rey"), ("time", "tiempo"), ("love", "amor")],
    "german": [("dog", "Hund"), ("cat", "Katze"), ("house", "Haus"), ("water", "Wasser"), ("bread", "Brot"), ("book", "Buch"),
               ("car", "Auto"), ("sun", "Sonne"), ("moon", "Mond"), ("tree", "Baum"), ("apple", "Apfel"), ("friend", "Freund"),
               ("day", "Tag"), ("night", "Nacht"), ("red", "rot"), ("green", "grün"), ("black", "schwarz"), ("white", "weiß"),
               ("man", "Mann"), ("woman", "Frau"), ("child", "Kind"), ("city", "Stadt"), ("school", "Schule"),
               ("cheese", "Käse"), ("milk", "Milch"), ("fish", "Fisch"), ("bird", "Vogel"), ("horse", "Pferd"),
               ("door", "Tür"), ("window", "Fenster"), ("sea", "Meer"), ("king", "König"), ("time", "Zeit"), ("love", "Liebe")],
}


@family
def translation(tok, rng):
    """Translate an English word in a few-shot list."""
    out = []
    for lang, pairs in TRANSLATIONS.items():
        v = Variant(lang, f"Word translation English to {lang.capitalize()}: after two example pairs, the translation of a new word.")
        for _ in range(400):
            (e1, f1), (e2, f2), (e, f), (e_, f_) = rng.sample(pairs, 4)
            t = "English: {e1}\n{L}: {f1}\n\nEnglish: {e2}\n{L}: {f2}\n\nEnglish: {e}\n{L}:"
            v.items.append(Item(t.format(e1=e1, f1=f1, e2=e2, f2=f2, e=e, L=lang.capitalize()), " " + f,
                                t.format(e1=e1, f1=f1, e2=e2, f2=f2, e=e_, L=lang.capitalize()), " " + f_))
        out.append(v)
    return out


POS_ADJ = "wonderful excellent amazing delightful fantastic superb lovely brilliant great perfect".split()
NEG_ADJ = "terrible awful horrible dreadful disappointing boring bad poor disgusting mediocre".split()


@family
def sentiment(tok, rng):
    """Few-shot sentiment labels."""
    out = []
    for vname, things in [("movie", ["movie", "film", "plot", "acting", "ending"]), ("restaurant", ["food", "service", "soup", "pizza", "dessert"])]:
        v = Variant(vname, f"Sentiment classification ({vname} reviews): after two labeled reviews, the label of a new review; the counterfactual swaps its adjective's polarity.")
        for _ in range(128):
            t1, t2, t3 = rng.sample(things, 3)
            shots = f"Review: The {t1} was {rng.choice(POS_ADJ)}.\nSentiment: positive\n\nReview: The {t2} was {rng.choice(NEG_ADJ)}.\nSentiment: negative\n\n"
            p, n = rng.choice(POS_ADJ), rng.choice(NEG_ADJ)
            if tok.count(" " + p) != tok.count(" " + n):
                continue
            a = Item(shots + f"Review: The {t3} was {p}.\nSentiment:", " positive", shots + f"Review: The {t3} was {n}.\nSentiment:", " negative")
            if rng.random() < 0.5:
                a = Item(a.cf_prefix, a.cf_answer, a.prefix, a.answer)
            v.items.append(a)
        out.append(v)
    return out


RHYMES = [["cat", "hat", "bat", "mat", "rat", "sat", "fat", "flat"], ["ring", "king", "sing", "wing", "thing", "bring", "spring"],
          ["light", "night", "bright", "fight", "sight", "right", "kite", "white"], ["day", "way", "play", "say", "stay", "gray", "may"],
          ["cake", "lake", "make", "bake", "take", "snake", "wake"], ["tree", "bee", "free", "see", "sea", "key", "knee"],
          ["moon", "soon", "spoon", "tune", "noon", "June"], ["rain", "train", "brain", "chain", "pain", "plain", "main"],
          ["book", "look", "cook", "hook", "took", "shook"], ["ball", "call", "fall", "tall", "wall", "hall", "small"],
          ["dog", "frog", "log", "fog", "jog"], ["bed", "red", "head", "bread", "said", "fed"], ["hand", "sand", "band", "land", "stand"],
          ["star", "car", "far", "jar", "bar"], ["fun", "sun", "run", "bun", "one", "done"], ["boat", "coat", "goat", "note", "float"]]


@family
def rhyme(tok, rng):
    """A rhyming word in a few-shot list."""
    v = Variant("pairs", "Rhyme: after example rhyming pairs, a word that rhymes with the given word (any member of its rhyme set counts).")
    for _ in range(128):
        g1, g2, g3, g4 = rng.sample(RHYMES, 4)
        shots = f"Words that rhyme:\n{g1[0]}: {g1[1]}\n{g2[0]}: {g2[1]}\n"
        w, w2 = g3[0], g4[0]
        ok, ok2 = [" " + x for x in g3[1:]], [" " + x for x in g4[1:]]
        v.items.append(Item(shots + f"{w}:", ok[0], shots + f"{w2}:", ok2[0], accept=ok, cf_accept=ok2))
    return [v]


GENDER_PAIRS = [("king", "queen"), ("man", "woman"), ("boy", "girl"), ("father", "mother"), ("brother", "sister"), ("son", "daughter"),
                ("uncle", "aunt"), ("husband", "wife"), ("prince", "princess"), ("actor", "actress"), ("nephew", "niece"),
                ("grandfather", "grandmother"), ("waiter", "waitress"), ("lord", "lady"), ("gentleman", "lady"), ("he", "she"),
                ("him", "her"), ("his", "her"), ("mr", "mrs"), ("bull", "cow"), ("rooster", "hen"), ("emperor", "empress"),
                ("god", "goddess"), ("hero", "heroine"), ("groom", "bride"), ("monk", "nun"), ("wizard", "witch"), ("duke", "duchess"),
                ("stepfather", "stepmother"), ("boyfriend", "girlfriend"), ("male", "female"), ("dad", "mom"), ("grandson", "granddaughter"),
                ("sir", "madam"), ("stallion", "mare"), ("host", "hostess")]


@family
def gender_analogy(tok, rng):
    """The female counterpart of a male word."""
    return fact_variants(GENDER_PAIRS, [("few_shot", "brother: sister\nfather: mother\n{X}:"),
                                        ("analogy", "Man is to woman as {X} is to")], "Analogy: the female counterpart of a male word.")


COMPARATIVES = [("big", "bigger"), ("small", "smaller"), ("fast", "faster"), ("slow", "slower"), ("tall", "taller"), ("short", "shorter"),
                ("old", "older"), ("young", "younger"), ("hot", "hotter"), ("cold", "colder"), ("long", "longer"), ("strong", "stronger"),
                ("weak", "weaker"), ("rich", "richer"), ("poor", "poorer"), ("good", "better"), ("bad", "worse"), ("happy", "happier"),
                ("easy", "easier"), ("heavy", "heavier"), ("light", "lighter"), ("dark", "darker"), ("high", "higher"), ("low", "lower"),
                ("wide", "wider"), ("deep", "deeper"), ("cheap", "cheaper"), ("clean", "cleaner"), ("loud", "louder"), ("quiet", "quieter"),
                ("near", "nearer"), ("far", "farther"), ("smart", "smarter"), ("hard", "harder"), ("soft", "softer"), ("sweet", "sweeter"),
                ("warm", "warmer"), ("cool", "cooler"), ("thin", "thinner"), ("thick", "thicker"), ("busy", "busier"), ("funny", "funnier"),
                ("large", "larger"), ("nice", "nicer"), ("safe", "safer"), ("brave", "braver"), ("late", "later"), ("early", "earlier")]


@family
def comparative(tok, rng):
    """The comparative form of an adjective."""
    return fact_variants(COMPARATIVES, [("few_shot", "tall: taller\nfast: faster\n{X}:"),
                                        ("sentence", "This box is {X}, but that box is even")], "Morphology: the comparative form of an adjective.")


CATEGORIES = [("rose", "flower"), ("tulip", "flower"), ("daisy", "flower"), ("apple", "fruit"), ("banana", "fruit"), ("mango", "fruit"),
              ("carrot", "vegetable"), ("potato", "vegetable"), ("onion", "vegetable"), ("dog", "animal"), ("lion", "animal"),
              ("elephant", "animal"), ("eagle", "bird"), ("sparrow", "bird"), ("parrot", "bird"), ("salmon", "fish"), ("shark", "fish"),
              ("trout", "fish"), ("oak", "tree"), ("pine", "tree"), ("maple", "tree"), ("piano", "instrument"), ("violin", "instrument"),
              ("guitar", "instrument"), ("hammer", "tool"), ("wrench", "tool"), ("saw", "tool"), ("car", "vehicle"), ("truck", "vehicle"),
              ("bus", "vehicle"), ("red", "color"), ("blue", "color"), ("green", "color"), ("soccer", "sport"), ("tennis", "sport"),
              ("basketball", "sport"), ("iron", "metal"), ("copper", "metal"), ("gold", "metal"), ("shirt", "clothing"),
              ("jacket", "clothing"), ("sock", "clothing"), ("Mars", "planet"), ("Jupiter", "planet"), ("Venus", "planet"),
              ("chess", "game"), ("poker", "game"), ("python", "snake"), ("cobra", "snake"), ("ant", "insect"), ("beetle", "insect"),
              ("bee", "insect"), ("Paris", "city"), ("Tokyo", "city"), ("London", "city"), ("whale", "mammal"), ("dolphin", "mammal"),
              ("cow", "mammal"), ("English", "language"), ("French", "language"), ("Spanish", "language"), ("Monday", "day"),
              ("Tuesday", "day"), ("January", "month"), ("March", "month")]


@family
def category(tok, rng):
    """The category of an object."""
    return fact_variants(CATEGORIES, [("type_of", "The {X} is a type of"), ("few_shot", "cat: animal\nhammer: tool\n{X}:")],
                         "Hypernym: the category a named thing belongs to.")


@family
def uppercase(tok, rng):
    """Uppercase a word in a few-shot list."""
    ws = words(tok, NOUNS)
    v = Variant("few_shot", "Case conversion: after two examples, the uppercase form of a new word.")
    for _ in range(600):
        a, b, c, d = rng.sample(ws, 4)
        t = "{a} -> {A}\n{b} -> {B}\n{c} ->"
        v.items.append(Item(t.format(a=a, A=a.upper(), b=b, B=b.upper(), c=c), " " + c.upper(), t.format(a=a, A=a.upper(), b=b, B=b.upper(), c=d), " " + d.upper()))
    return [v]


BIGRAMS = [("New", "York"), ("United", "States"), ("Los", "Angeles"), ("Hong", "Kong"), ("Sri", "Lanka"), ("Buenos", "Aires"),
           ("Saudi", "Arabia"), ("Puerto", "Rico"), ("Las", "Vegas"), ("San", "Francisco"), ("Costa", "Rica"), ("Prime", "Minister"),
           ("Vice", "President"), ("Supreme", "Court"), ("World", "War"), ("Middle", "East"), ("Wall", "Street"), ("White", "House"),
           ("Barack", "Obama"), ("Donald", "Trump"), ("Hillary", "Clinton"), ("Harry", "Potter"), ("Star", "Wars"), ("Tel", "Aviv"),
           ("Kuala", "Lumpur"), ("Real", "Madrid"), ("Manchester", "United"), ("Notre", "Dame"), ("Pearl", "Harbor"), ("Silicon", "Valley"),
           ("Abu", "Dhabi"), ("Rio", "de"), ("Pacific", "Ocean"), ("Atlantic", "Ocean"), ("Mount", "Everest"), ("Elon", "Musk"),
           ("Steve", "Jobs"), ("Bill", "Gates"), ("Vladimir", "Putin"), ("Angela", "Merkel"), ("Albert", "Einstein"),
           ("Isaac", "Newton"), ("Charles", "Darwin"), ("William", "Shakespeare"), ("Martin", "Luther"), ("Abraham", "Lincoln"),
           ("George", "Washington"), ("Winston", "Churchill"), ("Nelson", "Mandela"), ("Mother", "Teresa"), ("Wikipedia", "article"),
           ("Air", "Force"), ("Bank", "of"), ("Red", "Cross"), ("Coca", "Cola"), ("Burger", "King"), ("Taylor", "Swift"),
           ("Lady", "Gaga"), ("Michael", "Jackson"), ("Elvis", "Presley"), ("Bob", "Dylan"), ("Pink", "Floyd"), ("Rolling", "Stones")]


@family
def bigram(tok, rng):
    """Complete a frequent two-word name."""
    out = []
    for i, (vname, t) in enumerate([("sentence", "Last year I read a long article about {X}"), ("list", "Topics: weather, sports, {X}")]):
        v = Variant(vname, "Frequent bigram: the second word of a common multiword name follows its first word.")
        for a, b in BIGRAMS:
            v.items.append(Item(t.format(X=a), " " + b, tmpl=i))
        out.append(v)
    return out


IDIOMS = [("as well", "as"), ("in order", "to"), ("on the other", "hand"), ("at the same", "time"), ("in addition", "to"),
          ("for the first", "time"), ("as a matter of", "fact"), ("in front", "of"), ("as soon", "as"), ("in spite", "of"),
          ("at least", "one"), ("by the", "way"), ("in terms", "of"), ("due", "to"), ("according", "to"), ("rather", "than"),
          ("each", "other"), ("so far", "as"), ("as long", "as"), ("on behalf", "of"), ("in case", "of"), ("instead", "of"),
          ("a lot", "of"), ("in the middle", "of"), ("at the end of the", "day"), ("once upon a", "time"), ("better late than", "never"),
          ("last but not", "least"), ("the United", "States"), ("ladies and", "gentlemen"), ("salt and", "pepper"),
          ("bread and", "butter"), ("rock and", "roll"), ("black and", "white"), ("pros and", "cons"), ("up and", "down"),
          ("back and", "forth"), ("more or", "less"), ("sooner or", "later"), ("trial and", "error"), ("safe and", "sound"),
          ("first come, first", "served"), ("over and", "over"), ("give or", "take"), ("now and", "then"), ("here and", "there"),
          ("peace and", "quiet"), ("law and", "order"), ("time and", "again"), ("sick and", "tired"), ("again and", "again"),
          ("by and", "large"), ("odds and", "ends"), ("all of a", "sudden"), ("in the long", "run"), ("for the time", "being"),
          ("on the", "contrary"), ("with respect", "to"), ("in accordance", "with"), ("as far", "as"), ("in charge", "of"),
          ("on top", "of"), ("by means", "of"), ("in favor", "of")]


@family
def idiom(tok, rng):
    """Complete a fixed multiword expression."""
    out = []
    for i, (vname, t) in enumerate([("sentence", "She told me that, {X}"), ("start", "{X}")]):
        v = Variant(vname, "Fixed expression: the last word of a frequent multiword expression.")
        for a, b in IDIOMS:
            v.items.append(Item(t.format(X=a if i == 0 else a[0].upper() + a[1:]), " " + b, tmpl=i))
        out.append(v)
    return out


HTML_TAGS = "div span p table li ul ol strong em section header footer button form label td tr h1 h2 nav".split()


@family
def html_close(tok, rng):
    """Close the innermost open HTML tag."""
    v1 = Variant("nested", "HTML tag closing: after an inner element is closed, the next closing tag names the outer element.")
    v2 = Variant("text", "HTML tag closing: after text inside an element and an opening '</', the tag name of that element.")
    for _ in range(200):
        a, b, a2 = rng.sample(HTML_TAGS, 3)
        txt = rng.choice(["Hello", "Contact us", "Read more", "Home", "Price", "Next page", "Login"])
        v1.items.append(Item(f'<{a} class="box"><{b}>{txt}</{b}></', a, f'<{a2} class="box"><{b}>{txt}</{b}></', a2))
        v2.items.append(Item(f'<{a}>{txt}</', a, f'<{a2}>{txt}</', a2))
    return [v1, v2]

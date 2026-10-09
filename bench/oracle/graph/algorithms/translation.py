"""Word translation: after example pairs, the new English word in the language the last line names.
The tables are what the model recalls."""

FRENCH = {
    'apple': ' pomme', 'black': ' noir', 'book': ' livre', 'bread': ' pain', 'car': ' voiture',
    'cat': ' chat', 'cheese': ' fromage', 'child': ' enfant', 'city': ' ville', 'day': ' jour',
    'dog': ' chien', 'door': ' porte', 'fish': ' poisson', 'friend': ' ami', 'green': ' vert',
    'horse': ' cheval', 'house': ' maison', 'king': ' roi', 'love': ' amour', 'man': ' homme',
    'milk': ' lait', 'moon': ' lune', 'night': ' nuit', 'red': ' rouge', 'school': ' école',
    'sea': ' mer', 'sun': ' soleil', 'time': ' temps', 'tree': ' arbre', 'water': ' eau',
    'white': ' blanc', 'window': ' fenêtre', 'woman': ' femme',
}
GERMAN = {
    'apple': ' Apfel', 'bird': ' Vogel', 'black': ' schwarz', 'book': ' Buch', 'bread': ' Brot',
    'car': ' Auto', 'cat': ' Katze', 'cheese': ' Käse', 'child': ' Kind', 'city': ' Stadt',
    'day': ' Tag', 'dog': ' Hund', 'door': ' Tür', 'fish': ' Fisch', 'friend': ' Freund',
    'green': ' grün', 'horse': ' Pferd', 'house': ' Haus', 'king': ' König', 'love': ' Liebe',
    'man': ' Mann', 'milk': ' Milch', 'moon': ' Mond', 'night': ' Nacht', 'red': ' rot',
    'school': ' Schule', 'sea': ' Meer', 'sun': ' Sonne', 'time': ' Zeit', 'tree': ' Baum',
    'water': ' Wasser', 'white': ' weiß', 'window': ' Fenster', 'woman': ' Frau',
}
SPANISH = {
    'apple': ' manzana', 'bird': ' pájaro', 'black': ' negro', 'book': ' libro', 'bread': ' pan',
    'car': ' coche', 'cat': ' gato', 'cheese': ' queso', 'child': ' niño', 'city': ' ciudad',
    'day': ' día', 'dog': ' perro', 'door': ' puerta', 'fish': ' pescado', 'friend': ' amigo',
    'green': ' verde', 'horse': ' caballo', 'house': ' casa', 'king': ' rey', 'love': ' amor',
    'man': ' hombre', 'milk': ' leche', 'moon': ' luna', 'night': ' noche', 'red': ' rojo',
    'school': ' escuela', 'sea': ' mar', 'sun': ' sol', 'time': ' tiempo', 'tree': ' árbol',
    'water': ' agua', 'white': ' blanco', 'window': ' ventana', 'woman': ' mujer',
}
TABLES = {"French": FRENCH, "German": GERMAN, "Spanish": SPANISH}


def word(tokens):
    # the English word of the last pair
    out = []
    for t in range(len(tokens)):
        text = "".join(tokens[: t + 1])
        out.append(text.rsplit("English:", 1)[1].split("\n")[0].strip() if "English:" in text else None)
    return out


def language(tokens):
    # the language the last line names ("French:")
    out = []
    for t in range(len(tokens)):
        line = "".join(tokens[: t + 1]).split("\n")[-1]
        out.append(line.split(":")[0].strip() if ":" in line and line.split(":")[0].strip() in TABLES else None)
    return out


def rest(full, text):
    # what is left of `full` after the longest beginning of it that the text ends with
    return full[max(n for n in range(len(full)) if text.endswith(full[:n])):]


def answer(tokens, word, language):
    # the word's translation, or what is left of it once its first tokens are written
    out = []
    for t, (w, lang) in enumerate(zip(word, language)):
        full = TABLES[lang].get(w) if lang and w else None
        out.append(None if full is None else rest(full, "".join(tokens[: t + 1])))
    return out

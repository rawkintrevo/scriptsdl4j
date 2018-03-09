

def getCharacters(scene):
    # setting = scene[0].strip()
    characters = []
    for l in scene:
        if ":" in l:
            characters.append(l.split(":")[0])
        characters = list(set(characters))
    characters2 = [c.split("[")[0] for c in characters]
    characters3 = [c.split(")")[1] if ")" in c else c for c in characters2]
    characters4 = [c.split(".")[1] if "." in c else c for c in characters3]
    characters5 = []
    for c in characters4:
        characters5 = characters5 + c.split("+")
    char_set = list(set(["".join([c for c in character if c.isupper()]) for character in characters5]))
    return char_set

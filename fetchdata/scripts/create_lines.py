# Initial thoughs

# 1. If someone is interupted, their sentance remains incomplete.
# 2. Any line containing a colon- split after the colon (do this on App side)
# 3. Do a word count, probably going to train a word2vec at least on just star trek scripts
# 4. load scenes, start at line 3
# 5. Skip lines starting with "["
# 6. regex away anything in between "(" and ")"
# 7. Need to come up with a way to "bind" things like La Forge -> La_Forge (e.g. N-Grams) or maybe not.
# 8. Handle "'"

OUTPUT_FILE = "/home/rawkintrevo/gits/scriptsdl4j/fetchdata/scripts/corpus.txt"
EPISODE_DIR = "/home/rawkintrevo/gits/scriptsdl4j/fetchdata/scripts/star-trek-tng"

def stitchLines(lines, start, end):
    return " ".join(lines[start:end])

import os

with open(OUTPUT_FILE, "w") as fout:
    fout.write("")

for fname in os.listdir(EPISODE_DIR):
    fname_full = "/home/rawkintrevo/gits/scriptsdl4j/fetchdata/scripts/star-trek-tng/" + fname
    with open(fname_full, "r") as fin:
        lines = fin.readlines()
    lines = [l.replace("\n", "").strip() for l in lines]
    start_line = 0
    outlines = []
    c = 0
    for i in range(0, len(lines)):
        if lines[i].strip().startswith(">"):    # The title break
            start_line = i+1
        if lines[i].strip().startswith("["):
            outlines.append(stitchLines(lines, start_line, i))
            start_line = i+1
            outlines.append(lines[i].replace("\n", ""))
        if ":" in lines[i]:
            outlines.append(stitchLines(lines, start_line, i))
            start_line = i
    outlines2 = [l.split(':')[l.count(":")].strip() for l in outlines if ":" in l]
    with open(OUTPUT_FILE, "aw") as fout:
        fout.write("\n".join(outlines2))
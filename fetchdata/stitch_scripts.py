
import os

PATH = "scripts/star-trek-tng"

files = os.listdir(PATH)

with open(PATH + "/all-episodes.txt", 'wb') as fout:
    for fname in files:
        with open(PATH + "/" + fname, 'rb') as fin:
            fout.write("\nTITLE:\n")
            fout.write(fin.read())

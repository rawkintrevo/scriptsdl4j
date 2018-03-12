
import os

MIN_OCCURANCES_FOR_RECURRENT_CHAR = 8

print("Seeking characters that occur in more than %i scenes" % MIN_OCCURANCES_FOR_RECURRENT_CHAR)
all_chars = {}
all_topics = {}
base_scenes_dirs = ["/home/rawkintrevo/gits/scriptsdl4j/fetchdata/scenes/star-trek-tng"]
base_scripts_dir = "/home/rawkintrevo/gits/scriptsdl4j/fetchdata/scripts/star-trek-tng"
for targetdir in base_scenes_dirs:
    for f in os.listdir(targetdir):
        chars = []
        # topics = []
        with open(targetdir + "/" + f, "r") as fin:
            input = fin.readlines()
            chars += [c.replace("\n", "") for c in input[0].split(' ')]
            # topics += [c.replace("\n", "") for c in input[1].split(' ')]
            all_chars.update({chars[i] : all_chars.get(chars[i], 0) + 1 for i in range(0, len(chars))})
            # all_topics.update({topics[i] : all_topics.get(topics[i], 0) + 1 for i in range(0, len(topics))})


characters = [k for k,v in all_chars.iteritems() if v > MIN_OCCURANCES_FOR_RECURRENT_CHAR]

with open("/home/rawkintrevo/gits/scriptsdl4j/fetchdata/scripts/tng-all-chars.txt", 'w') as f:
    f.write("\n".join(characters))

print("Successfully wrote %i characters to file" % len(characters))
